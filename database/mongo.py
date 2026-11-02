from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError
from config import MONGODB_URI,MONGODB_DATABASE
import streamlit as st
@st.cache_resource
def client():
    if not MONGODB_URI: return None
    c=MongoClient(MONGODB_URI,serverSelectionTimeoutMS=4000); c.admin.command("ping"); return c
def db():
    c=client()
    return c[MONGODB_DATABASE] if c is not None else None
def get_db_status():
    try:
        d=db()
        if d is None: return False,"MONGODB_URI missing"
        d.command("ping"); return True,"Connected"
    except Exception as e: return False,str(e)
def save_detection(record):
    try:
        d=db()
        if d is None: return False,"MongoDB URI not configured"
        d.detections.insert_one(dict(record)); return True,"Inserted"
    except PyMongoError as e: return False,str(e)
def fetch_detections(limit=1000):
    try:
        d=db()
        if d is None: return []
        return list(d.detections.find({},{"_id":0}).sort("created_at",DESCENDING).limit(limit))
    except PyMongoError: return []
