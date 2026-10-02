from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()
db = MongoClient(os.getenv("MONGO_URI"))["biosentry_db"]

print("Chargement...")
df = pd.DataFrame(db["ml_predictions"].find({}, {"_id": 0}))

df["nb_algos"] = (
    df["lof_anomaly"].astype(int) +
    df["iforest_anomaly"].astype(int) +
    df["zscore_anomaly"].astype(int)
)

def get_alert(n):
    if n == 3: return "CRITIQUE"
    if n == 2: return "ELEVEE"
    if n == 1: return "MODEREE"
    return "NORMAL"

df["alert_level"] = df["nb_algos"].apply(get_alert)

# Mise à jour en BULK — beaucoup plus rapide !
from pymongo import UpdateOne
print("Mise à jour bulk...")
operations = []
for _, row in df.iterrows():
    operations.append(UpdateOne(
        {"drug": row["drug"], "side_effect": row["side_effect"], "period": row["period"]},
        {"$set": {"nb_algos": int(row["nb_algos"]), "alert_level": row["alert_level"]}}
    ))

# Envoie tout en une seule fois
db["ml_predictions"].bulk_write(operations, ordered=False)

print(f"✅ {len(df)} documents mis à jour !")
print(f"  🟢 NORMAL   : {db['ml_predictions'].count_documents({'alert_level':'NORMAL'})}")
print(f"  🟡 MODEREE  : {db['ml_predictions'].count_documents({'alert_level':'MODEREE'})}")
print(f"  🟠 ELEVEE   : {db['ml_predictions'].count_documents({'alert_level':'ELEVEE'})}")
print(f"  🔴 CRITIQUE : {db['ml_predictions'].count_documents({'alert_level':'CRITIQUE'})}")