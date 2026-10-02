from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
import time, random, re, os, argparse
from datetime import datetime, timezone

load_dotenv()

SOURCE_NAME   = "webmd"
SEARCH_URL    = "https://www.webmd.com/search?query={query}"
DELAY_DRUG    = (5.0, 12.0)
DELAY_403     = (45.0, 90.0)
RESTART_EVERY = 30
HEADLESS      = True

_driver = None


def get_db():
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise EnvironmentError("MONGO_URI manquant dans .env")
    return MongoClient(uri)["biosentry_db"]


def load_drug_list():
    db   = get_db()
    docs = list(db["drug_list"].find({}, {"_id": 0, "name": 1, "category": 1, "synonyms": 1}))
    seen, result = set(), []
    for d in docs:
        name = d.get("name", "").strip().lower()
        if name and name not in seen:
            seen.add(name)
            result.append({"drug_name": name, "category": d.get("category", ""), "synonyms": d.get("synonyms", [])})
    print(f"[DRUG_LIST] {len(result)} médicaments")
    return result


def get_already_done():
    return set(get_db()["raw_sources"].distinct("drug_name", {"source": SOURCE_NAME}))


def ensure_indexes():
    col = get_db()["raw_sources"]
    try:
        col.drop_index("webmd_drug_unique")
    except Exception:
        pass
    col.create_index(
        [("drug_name", 1), ("effect_type", 1)],
        unique=True,
        partialFilterExpression={"source": SOURCE_NAME},
        name="webmd_drug_unique",
    )
    for field in ["drug_name", "source", "category", "scraped_at"]:
        col.create_index(field)


def save_to_mongo(records):
    if not records:
        return
    col = get_db()["raw_sources"]
    ops = [
        UpdateOne(
            {"source": SOURCE_NAME, "drug_name": r["drug_name"], "effect_type": r["effect_type"]},
            {"$setOnInsert": r},
            upsert=True,
        )
        for r in records
    ]
    res = col.bulk_write(ops, ordered=False)
    print(f"  [MONGO] Insérés: {res.upserted_count} | Existants: {res.matched_count}")


def make_driver():
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    return driver


def get_driver():
    global _driver
    if _driver is None:
        _driver = make_driver()
    return _driver


def restart_driver():
    global _driver
    if _driver:
        try:
            _driver.quit()
        except Exception:
            pass
        _driver = None
    time.sleep(2)
    get_driver()


def human_delay(mn=0.8, mx=2.5):
    time.sleep(random.uniform(mn, mx))


def human_scroll(driver):
    for _ in range(random.randint(2, 4)):
        driver.execute_script(f"window.scrollBy(0, {random.randint(200, 500)});")
        time.sleep(random.uniform(0.3, 0.7))


def dismiss_cookie_banner(driver):
    for sel in ["button#onetrust-accept-btn-handler", "button.onetrust-close-btn-handler",
                "button[title='Accept All Cookies']", "[aria-label='Close']"]:
        try:
            WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.CSS_SELECTOR, sel))).click()
            human_delay(0.5, 1.0)
            return
        except Exception:
            continue


def _pick_best_drug_link(driver, drug_name):
    name_main = drug_name.lower().split()[0]
    best = fallback = None
    for link in driver.find_elements(By.CSS_SELECTOR, "a[href*='/drugs/']"):
        href = (link.get_attribute("href") or "").lower()
        text = link.text.strip().lower()
        if any(x in href for x in ["/drugs/2/", "/drugs/index", "search?", "#"]):
            continue
        if not re.search(r"/drugs/[a-z]", href):
            continue
        if name_main in href or name_main in text:
            return link
        if fallback is None:
            fallback = link
    return best or fallback


def _search_and_click(driver, search_name, drug_name):
    try:
        driver.get(SEARCH_URL.format(query=search_name.replace(" ", "+")))
        human_delay(2.5, 5.0)
        dismiss_cookie_banner(driver)
        human_scroll(driver)
        human_delay(1.0, 2.0)
        try:
            WebDriverWait(driver, 12).until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/drugs/']")))
        except TimeoutException:
            return None
        link_elem = _pick_best_drug_link(driver, drug_name)
        if not link_elem:
            return None
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link_elem)
        human_delay(0.6, 1.4)
        try:
            link_elem.click()
        except ElementClickInterceptedException:
            driver.execute_script("arguments[0].click();", link_elem)
        human_delay(3.0, 6.0)
        final_url = driver.current_url.split("#")[0]
        return (final_url + "#sideeffects") if "/drugs/" in final_url else None
    except Exception as e:
        print(f"    [SELENIUM-ERR] {e}")
        return None


def find_webmd_url_selenium(drug_name, synonyms):
    driver     = get_driver()
    candidates = [drug_name] + [s for s in synonyms if len(s) < 30]
    for name in candidates[:4]:
        url = _search_and_click(driver, name, drug_name)
        if url:
            return url
        human_delay(1.5, 3.5)
    return None


def _is_blocked(html):
    return any(s in html.lower() for s in ["access denied", "403 forbidden", "captcha", "unusual traffic", "you are a robot"]) and len(html) < 8000


def _click_sideeffects_tab(driver):
    for sel in ["a[href='#sideeffects']", "a[href*='sideeffects']", "li a[data-tab='sideeffects']",
                "//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'side effect')]"]:
        try:
            by  = By.XPATH if sel.startswith("//") else By.CSS_SELECTOR
            tab = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((by, sel)))
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", tab)
            human_delay(0.4, 0.9)
            tab.click()
            human_delay(1.5, 3.0)
            return True
        except Exception:
            continue
    return False


def _extract_effects_from_dom(driver):
    result = {"common": [], "serious": [], "rare": []}
    try:
        data = driver.execute_script("""
            const containers = [
                document.querySelector('#sideeffects .monograph-content'),
                document.querySelector('#sideeffects .side-effects-container'),
                document.querySelector('div.monograph-content'),
                document.querySelector('#sideeffects'),
            ];
            const container = containers.find(c => c !== null);
            if (!container) return [];
            const nodes = Array.from(container.querySelectorAll('h2,h3,h4,li'));
            let started = false;
            const results = [];
            for (const node of nodes) {
                const text = node.innerText.trim();
                const textLow = text.toLowerCase();
                if (!text) continue;
                const isH = ['H2','H3','H4'].includes(node.tagName);
                if (!started && isH && textLow.includes('side effect')) started = true;
                if (!started) continue;
                if (node.tagName === 'H2' && !textLow.includes('side effect')) break;
                if (isH) {
                    let type = 'keep';
                    if (node.tagName !== 'H4') {
                        if (textLow.includes('serious') || textLow.includes('severe')) type = 'serious';
                        else if (textLow.includes('rare') || textLow.includes('less common')) type = 'rare';
                        else if (textLow.includes('side effect')) type = 'common';
                    }
                    results.push({ tag: 'h', type });
                } else if (node.tagName === 'LI' && text.length < 250) {
                    results.push({ tag: 'li', text });
                }
            }
            return results;
        """)
        if not data:
            return result
        current_type, seen = "common", set()
        for el in data:
            if el["tag"] == "h":
                if el["type"] != "keep":
                    current_type = el["type"]
            elif el["text"] and el["text"] not in seen:
                seen.add(el["text"])
                result[current_type].append(el["text"])
    except Exception as e:
        print(f"    [DOM-ERR] {e}")
    return result


def extract_side_effects_selenium(url):
    driver = get_driver()
    result = {"common": [], "serious": [], "rare": []}
    try:
        driver.get(url.split("#")[0])
        human_delay(2.5, 4.5)
        dismiss_cookie_banner(driver)
        if _is_blocked(driver.page_source):
            time.sleep(random.uniform(*DELAY_403))
            restart_driver()
            return result
        _click_sideeffects_tab(driver)
        content_elem = None
        for sel in ["#sideeffects .monograph-content", "#sideeffects .side-effects-container .active-tab .monograph-content",
                    "div.monograph-content", ".side-effects-container"]:
            try:
                content_elem = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
                break
            except TimeoutException:
                continue
        if not content_elem:
            return result
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", content_elem)
        human_delay(1.5, 3.0)
        result = _extract_effects_from_dom(driver)
    except Exception as e:
        print(f"  [EXTRACT-ERR] {e}")
    return result


def scrape_drug(drug_name, category, synonyms):
    url = find_webmd_url_selenium(drug_name, synonyms)
    if not url:
        return []
    effects = extract_side_effects_selenium(url)
    if not any(effects.values()):
        return []
    now, records = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), []
    for effect_type, items in effects.items():
        if items:
            records.append({
                "source"     : SOURCE_NAME,
                "category"   : category,
                "drug_name"  : drug_name,
                "scraped_at" : now,
                "url"        : url,
                "confidence" : "high" if effect_type == "serious" else "medium",
                "effect_type": effect_type,
                "effects"    : items,
            })
    return records


def print_stats():
    col = get_db()["raw_sources"]
    print(f"\n[STATS] Total: {col.count_documents({'source': SOURCE_NAME})} | "
          f"Médicaments: {len(col.distinct('drug_name', {'source': SOURCE_NAME}))}")
    for et in ["common", "serious", "rare"]:
        print(f"  {et}: {col.count_documents({'source': SOURCE_NAME, 'effect_type': et})}")


def run(reset=False, test_drug=None):
    global _driver
    db = get_db()
    if reset:
        db["raw_sources"].delete_many({"source": SOURCE_NAME})
    ensure_indexes()

    if test_drug:
        try:
            records = scrape_drug(test_drug, "TEST", [])
            if records:
                save_to_mongo(records)
        finally:
            if _driver:
                _driver.quit()
        return

    target_drugs = load_drug_list()
    if not target_drugs:
        return

    already_done = get_already_done()
    remaining    = [d for d in target_drugs if d["drug_name"] not in already_done]
    print(f"  {len(already_done)}/{len(target_drugs)} complétés | {len(remaining)} restants")

    try:
        for i, drug in enumerate(remaining, start=len(already_done) + 1):
            idx = i - len(already_done)
            if idx > 1 and (idx - 1) % RESTART_EVERY == 0:
                restart_driver()
            print(f"[{i:>3}/{len(target_drugs)}] {drug['drug_name']}")
            records = scrape_drug(drug["drug_name"], drug["category"], drug["synonyms"])
            save_to_mongo(records)
            time.sleep(random.uniform(*DELAY_DRUG))
    finally:
        if _driver:
            _driver.quit()
            _driver = None

    print_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",   action="store_true")
    parser.add_argument("--test",    type=str, default=None)
    parser.add_argument("--visible", action="store_true")
    args = parser.parse_args()
    if args.visible:
        HEADLESS = False
    run(reset=args.reset, test_drug=args.test)
