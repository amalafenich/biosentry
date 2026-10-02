"""
Bio-Sentry — MedlinePlus Drug Scraper (v2 — Final)
====================================================
Corrections v2 :
  - FIX 1 : Suppression § et symboles parasites dans brand_names
  - FIX 2 : Side effects extraits aussi depuis <p> (pas seulement <li>)
  - FIX 3 : Warnings corrigés → bon sélecteur id="section-warning"

Mode TEST  : LIMIT = 20  → scrape 20 médicaments
Mode FULL  : LIMIT = None → scrape tous les médicaments A→Z

Usage :
    pip install selenium webdriver-manager beautifulsoup4 tqdm
    python biosentry_medlineplus_scraper_v2.py

⚠️  Les drug monographs MedlinePlus sont © ASHP.
    Usage strictement limité à la recherche académique.
"""

import re
import json
import time
from tqdm import tqdm
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════

LIMIT                  = None     # ← 20 pour tester | None pour tout A→Z
OUTPUT_FILE            = "biosentry_medlineplus_full.json"
BASE_URL               = "https://medlineplus.gov"
WAIT_TIMEOUT           = 10      # secondes max pour attendre le chargement
SLEEP_BETWEEN_REQUESTS = 1.5     # secondes entre chaque drug


# ══════════════════════════════════════════════
# SELENIUM SETUP
# ══════════════════════════════════════════════

def create_driver() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=BioSentry-Research/1.0 (pharmacovigilance; contact@biosentry.com)"
    )
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    return driver


# ══════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Nettoie un texte brut extrait du HTML."""
    if not text:
        return ""
    text = re.sub(r"http\S+", "", text)           # supprimer URLs
    text = re.sub(r"\s+", " ", text)              # normaliser espaces
    text = text.replace("\u00ae", "")             # ® unicode
    text = text.replace("\u2122", "")             # ™ unicode
    text = text.replace("\u00b6", "")             # ¶
    text = text.replace("\u00a7", "")             # § unicode (FIX 1)
    text = text.replace("\u2020", "")             # † unicode
    text = text.replace("\u2021", "")             # ‡ unicode
    # Supprimer titres parasites résiduels
    noise_patterns = [
        r"Why is this medication prescribed\?\s*",
        r"What side effects can this medication cause\?\s*",
        r"Brand names\s*(Expand Section\s*)?",
        r"Expand Section\s*",
        r"More common side effects\s*",
        r"Less common side effects\s*",
        r"Rare side effects\s*",
    ]
    for p in noise_patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text.strip(" -•:\n\t")


def remove_duplicates(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def wait_for_section(driver, section_id: str):
    """Attend qu'une section soit présente dans le DOM."""
    try:
        WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.ID, section_id))
        )
    except Exception:
        pass  # Si timeout, on continue avec ce qui est dispo


# ══════════════════════════════════════════════
# EXTRACTEURS PAR SECTION
# ══════════════════════════════════════════════

def get_drug_name(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    return clean_text(h1.get_text(strip=True)) if h1 else ""


def get_prescribed_for(soup: BeautifulSoup) -> list:
    """Extrait les indications thérapeutiques (section 'why')."""
    section = soup.find("div", id="why")
    if not section:
        return []

    useless = [
        "ask your doctor", "talk to your doctor",
        "this medication may be prescribed",
        "for more information"
    ]
    results = []
    for p in section.find_all("p"):
        text = clean_text(p.get_text(" ", strip=True))
        if len(text) < 40:
            continue
        if any(u in text.lower() for u in useless):
            continue
        results.append(text)

    return remove_duplicates(results)


def get_side_effects(soup: BeautifulSoup) -> dict:
    """
    Extrait les effets secondaires en distinguant :
      - mild   : effets courants / bénins
      - serious: effets graves
    FIX 2 : parse aussi les <p> quand il n'y a pas de <li>
    Chaque <li> est splitté sur ";" pour isoler chaque symptôme.
    """
    section = soup.find("div", id="side-effects")
    if not section:
        return {"mild": [], "serious": []}

    useless_phrases = [
        "tell your doctor", "call your doctor", "if you experience",
        "medwatch", "emergency medical treatment",
        "doctor immediately", "serious side effect", "1-800",
        "food and drug administration", "fda", "call your"
    ]

    results          = {"mild": [], "serious": []}
    current_category = "mild"

    for tag in section.find_all(["h3", "ul", "p"]):

        # ── Détecter la catégorie via le titre <h3>
        if tag.name == "h3":
            h3_text = tag.get_text().lower()
            current_category = "serious" if "serious" in h3_text else "mild"

        # ── Extraire depuis les listes <ul><li>
        elif tag.name == "ul":
            for li in tag.find_all("li"):
                raw = clean_text(li.get_text(" ", strip=True))
                if any(u in raw.lower() for u in useless_phrases):
                    continue
                # Split sur ";" → plusieurs symptômes dans un <li>
                for symptom in re.split(r";", raw):
                    symptom = symptom.strip(" .,")
                    if 3 < len(symptom) and len(symptom.split()) <= 15:
                        results[current_category].append(symptom)

        # ── FIX 2 : Extraire depuis <p> quand pas de <li>
        elif tag.name == "p":
            raw = clean_text(tag.get_text(" ", strip=True))
            if any(u in raw.lower() for u in useless_phrases):
                continue
            # Seulement les <p> courts = vrais symptômes
            if len(raw.split()) <= 30 and len(raw) > 5:
                for symptom in re.split(r"[;,]", raw):
                    symptom = symptom.strip(" .,")
                    if 3 < len(symptom) and len(symptom.split()) <= 10:
                        results[current_category].append(symptom)

    results["mild"]    = remove_duplicates(results["mild"])
    results["serious"] = remove_duplicates(results["serious"])
    return results


def get_brand_names(soup: BeautifulSoup) -> list:
    """
    Extrait les noms commerciaux (section 'brand-name-1').
    FIX 1 : suppression de §, ¶, †, ‡, * et autres symboles parasites.
    """
    section = soup.find("div", id="brand-name-1")
    if not section:
        return []

    brands = []
    for li in section.find_all("li"):
        text = clean_text(li.get_text(" ", strip=True))
        # FIX 1 — supprimer tous les symboles parasites résiduels
        text = re.sub(r"[§¶†‡*]", "", text).strip()
        if len(text) >= 2:
            brands.append(text)

    return remove_duplicates(brands)


def get_drug_class(soup: BeautifulSoup) -> str:
    """
    Extrait la classe thérapeutique depuis la section 'why'.
    Pattern : "X is in a class of medications called Y"
    """
    section = soup.find("div", id="why")
    if not section:
        return ""
    first_p = section.find("p")
    if not first_p:
        return ""
    text = first_p.get_text(" ", strip=True)
    match = re.search(
        r"in a class of medications? called ([^.]+)\.",
        text, re.IGNORECASE
    )
    return match.group(1).strip() if match else ""


def get_warnings(soup: BeautifulSoup) -> list:
    """
    Extrait les avertissements importants (boxed warning).
    FIX 3 : bon sélecteur → id="section-warning" (pas "boxed-warning")
    """
    # FIX 3 — le vrai ID dans le HTML est "section-warning"
    section = soup.find("div", id="section-warning")
    if not section:
        return []

    warnings = []

    # Extraire les <li> du warning
    for li in section.find_all("li"):
        text = clean_text(li.get_text(" ", strip=True))
        if len(text) > 5:
            warnings.append(text)

    # Extraire aussi les <p> du warning (souvent plus informatifs)
    for p in section.find_all("p"):
        text = clean_text(p.get_text(" ", strip=True))
        if len(text) > 20:
            warnings.append(text[:300])  # limiter à 300 chars

    return remove_duplicates(warnings)


# ══════════════════════════════════════════════
# SCRAPE UN SEUL MÉDICAMENT
# ══════════════════════════════════════════════

def scrape_drug(driver, url: str) -> dict:
    """Charge la page d'un médicament et extrait toutes les données."""
    driver.get(url)

    # Attendre que la section principale soit chargée
    wait_for_section(driver, "side-effects")

    soup = BeautifulSoup(driver.page_source, "html.parser")

    side_effects = get_side_effects(soup)

    return {
        "url":            url,
        "drug_name":      get_drug_name(soup),
        "drug_class":     get_drug_class(soup),
        "brand_names":    get_brand_names(soup),
        "prescribed_for": get_prescribed_for(soup),
        "side_effects": {
            "mild":    side_effects["mild"],
            "serious": side_effects["serious"],
            "total":   len(side_effects["mild"]) + len(side_effects["serious"])
        },
        "warnings":       get_warnings(soup),
        "source":         "MedlinePlus",
        "scraped_at":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ══════════════════════════════════════════════
# COLLECTE DES LIENS — A → Z
# ══════════════════════════════════════════════

def get_all_drug_links(driver, limit: int = None) -> list:
    """
    Parcourt les pages index A→Z de MedlinePlus et collecte
    tous les liens /druginfo/meds/.
    Si limit=None → collecte tout A→Z.
    Si limit=N    → s'arrête dès que N liens sont collectés.
    """
    links = []
    seen  = set()
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    print("\n📋 Collecte des liens médicaments...")

    for letter in alphabet:
        index_url = f"{BASE_URL}/druginfo/drug_{letter}a.html"
        print(f"  → Lettre {letter} : {index_url}")

        driver.get(index_url)
        time.sleep(1.5)

        elems = driver.find_elements(By.TAG_NAME, "a")

        for elem in elems:
            href = elem.get_attribute("href")
            if href and "/druginfo/meds/" in href and href not in seen:
                seen.add(href)
                links.append(href)

        print(f"     {len(links)} liens collectés au total")

        # Arrêt anticipé si limite atteinte (FIX bug original)
        if limit and len(links) >= limit:
            links = links[:limit]
            print(f"\n  ✅ Limite de {limit} atteinte — arrêt collecte")
            break

    print(f"\n📦 Total liens : {len(links)}")
    return links


# ══════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  BIO-SENTRY — MedlinePlus Drug Scraper v2")
    print(f"  Mode    : {'TEST (' + str(LIMIT) + ' drugs)' if LIMIT else 'FULL (A→Z)'}")
    print(f"  Output  : {OUTPUT_FILE}")
    print("=" * 55)

    driver = create_driver()

    try:
        # 1. Collecter les URLs
        drug_links = get_all_drug_links(driver, limit=LIMIT)

        if not drug_links:
            print("❌ Aucun lien trouvé.")
            return

        # 2. Scraper chaque médicament
        all_data = []
        errors   = []

        print(f"\n🧪 Scraping {len(drug_links)} médicaments...\n")

        for url in tqdm(drug_links, desc="Scraping", unit="drug"):
            try:
                drug_data = scrape_drug(driver, url)
                all_data.append(drug_data)

                # Log rapide
                name      = drug_data.get("drug_name", url)
                n_effects = drug_data["side_effects"]["total"]
                n_warn    = len(drug_data["warnings"])
                print(f"  ✓ {name[:45]:<45} | effects={n_effects:>3} | warnings={n_warn}")

            except Exception as e:
                errors.append({"url": url, "error": str(e)})
                print(f"  ❌ Erreur : {url}")
                print(f"     {e}")

            time.sleep(SLEEP_BETWEEN_REQUESTS)

        # 3. Sauvegarder JSON principal
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)

        # 4. Résumé final
        print(f"\n{'=' * 55}")
        print(f"  ✅ Dataset sauvegardé : {OUTPUT_FILE}")
        print(f"  📊 Médicaments scrapés : {len(all_data)}")
        print(f"  ❌ Erreurs             : {len(errors)}")
        total_effects  = sum(d["side_effects"]["total"] for d in all_data)
        total_warnings = sum(len(d["warnings"]) for d in all_data)
        total_brands   = sum(len(d["brand_names"]) for d in all_data)
        print(f"  💊 Effets secondaires  : {total_effects}")
        print(f"  ⚠️  Warnings            : {total_warnings}")
        print(f"  🏷️  Noms commerciaux    : {total_brands}")
        print(f"{'=' * 55}")

        # 5. Sauvegarder les erreurs si besoin
        if errors:
            error_file = OUTPUT_FILE.replace(".json", "_errors.json")
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(errors, f, ensure_ascii=False, indent=4)
            print(f"  ⚠️  Erreurs log : {error_file}")

    finally:
        driver.quit()
        print("\n🔒 Driver fermé proprement.")


if __name__ == "__main__":
    main()
