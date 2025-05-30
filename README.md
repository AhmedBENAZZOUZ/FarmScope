# 🌾 FarmScope – Retrieval-Augmented Generation (RAG) for Smart Agriculture

FarmScope is a smart assistant designed to provide agronomic insights based on geospatial, satellite, and weather data. It leverages a Retrieval-Augmented Generation (RAG) pipeline to answer natural language queries from users in the agricultural domain.

---

## 🧠 Project Overview

FarmScope combines the power of:

- 🌍 Geolocation-based data (via geocode/maps)
- 🛰️ Satellite imagery (Sentinel-1 & Sentinel-2)
- 🌦️ Weather forecasts (via Open-Meteo)
- 🤖 LLMs (OpenAI, LangChain) for analysis & generation
- ⚡ FastAPI backend for serving RESTful endpoints

---

## 🔁 Pipeline Breakdown

| Step | Component | Description |
|------|-----------|-------------|
| 1️⃣ | `decide_action()` | Detect user intent using LLM + keyword fallback |
| 2️⃣ | Geocoding | Convert place names or coordinates to geo-points via [Maps.co](https://geocode.maps.co) |
| 3️⃣ | Satellite Data | Query Sentinel-2 (NDVI, NDWI, SAVI) & Sentinel-1 (VV for soil moisture) using GEE |
| 4️⃣ | Weather Data | Fetch daily weather from [Open-Meteo](https://open-meteo.com) |
| 5️⃣ | RAG Retrieval | Inject retrieved data directly into the LLM prompt (no vector DB used) |
| 6️⃣ | Generation | Use OpenAI LLM to generate structured markdown advice |
| 7️⃣ | Report Export | Output markdown report including all data + user context |
| 8️⃣ | API Interface | FastAPI endpoint `/api/analyze` for interaction |

---

## 🚀 Features

- 🌍 Natural Language Location Detection
- 🛰️ Cloud-masked satellite retrieval with 1km buffer
- 📊 Weather forecast retrieval (7-day)
- 🧠 Agronomic analysis with Tunisian context
- 📄 Downloadable markdown reports
- 🔁 Multi-turn user conversation support

---

## ⚙️ Technologies Used

| Tool | Purpose |
|------|--------|
| FastAPI | REST API backend |
| LangChain | LLM orchestration |
| OpenAI | LLM for generation & intent detection |
| Google Earth Engine | Satellite imagery processing |
| Geemap | Python API for GEE |
| Open-Meteo | Weather API |
| Markdown / HTML | Report formatting |
| Geocode API | Location parsing |

---

## 📦 Installation

```bash
git clone https://github.com/<your-org>/FarmScope.git
cd FarmScope
pip install -r requirements.txt
