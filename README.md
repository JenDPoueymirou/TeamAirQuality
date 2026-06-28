# TeamAirQuality
This is a project we all created for the Bloomberg Hackathon.
# NYC Air Pollution & Disease: A Borough-Level Analysis

> **Research Question:** Are neighborhoods with higher truck traffic and pollution levels associated with higher rates of asthma ER visits, cardiovascular hospitalizations, and pollution-related deaths across NYC's 5 boroughs?

---

## Project Overview

This project investigates the relationship between air pollution and disease outcomes across New York City's 5 boroughs from 2005 to 2024. Using data from four live APIs and nearly 20 years of historical records, the analysis examines how long-term exposure to PM2.5, NO2, and ozone — particularly in neighborhoods near highways, bridges, tunnels, and truck corridors — correlates with higher rates of:

- Asthma emergency department visits
- Cardiovascular hospitalizations
- Respiratory hospitalizations
- Cardiac and respiratory deaths

A key focus is the environmental justice dimension: communities in the South Bronx, Brownsville, Greenpoint, and Gowanus face compounding pollution burdens from truck traffic, industrial waterways, and Superfund sites that drive disease rates far above citywide averages.

---

## Data Sources & APIs

| # | Source | Dataset | Key Required |
|---|---|---|---|
| 1 | NYC Open Data | Air Quality & Health Impacts `c3uy-2p5r` | No |
| 2 | NYC Open Data | PM2.5 Attributable Asthma ED Visits `ebe7-6eah` | No |
| 3 | EPA AirNow | Real-time AQI by zip code | Yes (free) |
| 4 | PurpleAir | Community sensor network — fills Brooklyn & Manhattan gaps | Yes (free) |

**Total dataset: 19,261 rows across all sources**

---

## Variables

### Dependent Variable
| Variable | Type |
|---|---|
| Asthma ER visit rate (per 100,000) | Quantitative |

### Independent Variables (9 total)
| Variable | Type |
|---|---|
| PM2.5 concentration (mcg/m³) | Quantitative |
| Nitrogen dioxide NO2 (ppb) | Quantitative |
| Ozone O3 (ppb) | Quantitative |
| Annual truck vehicle miles traveled | Quantitative |
| Annual total vehicle miles traveled | Quantitative |
| Real-time AQI (EPA AirNow) | Quantitative |
| Real-time PM2.5 (PurpleAir) | Quantitative |
| Borough | Categorical |
| Time Period | Categorical |

---

## Project Structure

```
Pollution&DiseaseNYC/
├── src/
│   ├── dataingestion.py        # Pulls data from all 4 APIs → saves CSVs
│   └── datamerge.py            # Merges CSVs into merged_final.csv
├── Notebooks/
│   ├── final_project_main.ipynb     # Main analysis notebook
│   └── nyc_pollution_map.html  # Interactive zip code heatmap
├── Data/                       # CSVs saved here (not tracked by Git)
│   ├── Air_Quality_and_Health_Impacts.csv
│   ├── nyc_air_quality_health.csv
│   ├── asthma_ed_pm25.csv
│   ├── airnow_realtime_aqi.csv
│   ├── purpleair_pm25.csv
│   └── merged_final.csv
├── .env                        # API keys (not tracked by Git — see .env.example)
├── .env.example                # Template showing required keys
├── .gitignore
└── ReadMe.md
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/JenDPoueymirou/Pollution-DiseaseNYC.git
cd Pollution-DiseaseNYC
```

### 2. Create a conda environment
```bash
conda create -n school python=3.12
conda activate school
```

### 3. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn folium branca requests python-dotenv
```

### 4. Set up API keys
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
Then edit `.env`:
```
AIRNOW_API_KEY=your_airnow_key_here
PURPLEAIR_API_KEY=your_purpleair_read_key_here
CENSUS_API_KEY=your_census_key_here
```

**Getting free API keys:**
- **AirNow:** Register at [airnow.gov/api](https://docs.airnowapi.org/account/request/)
- **PurpleAir:** Register at [develop.purpleair.com](https://develop.purpleair.com)
- **Census Bureau:** Register at [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html)

### 5. Run data ingestion
```bash
python src/dataingestion.py
python src/datamerge.py
```

### 6. Open the notebook
Open `Notebooks/final_project_main.ipynb` in Jupyter or VSCode and run all cells.

---

## Key Findings

### Pollution Trends
- PM2.5 levels have declined across all 5 boroughs since 2008, but the **Bronx and Manhattan consistently exceed the EPA annual standard of 12 mcg/m³**
- NO2 levels are highest in Manhattan due to the density of tunnel approaches (Lincoln, Holland, Battery tunnels) and the FDR Drive
- Ozone levels are paradoxically lower directly on highways and higher in surrounding neighborhoods — residents living near the Cross Bronx Expressway face both direct exhaust and ozone drift

### Health Outcomes
- The **Bronx has the highest asthma ER visit rate** at 167 per 100,000 — nearly 7x higher than the lowest borough
- **Cardiovascular hospitalizations correlate strongly with PM2.5 deaths** (r = 0.88) — the two health outcomes move together
- **Respiratory hospitalizations correlate with cardiovascular hospitalizations** (r = 0.86) — pollution affects the heart and lungs simultaneously
- Notable hotspots: Hunts Point (10474), Brownsville (11212), Greenpoint (11222), Broadway Junction (11233), Gowanus (11215/11217)

### Real-Time Data
- Live AirNow readings show AQI averaging **36–41 across boroughs** at time of analysis
- PurpleAir community sensors reveal **hyperlocal spikes** near truck corridors not captured by official monitors
- Brooklyn and Manhattan have the fewest official air monitors despite having significant pollution sources — a finding that itself reflects environmental justice disparities

---

## Interactive Map

The file `Notebooks/nyc_pollution_map.html` contains a standalone interactive heatmap of all NYC zip codes. Open it in any browser.

**Features:**
- Blue → red color scale (blue = cleaner, red = more polluted)
- Hover over any zip code for AQI, PM2.5, asthma ER rate, cardiovascular hospitalization rate, truck traffic index, and infrastructure notes
- Click any zip code to pin its data
- Special callouts for Superfund sites, major highway corridors, and truck terminals

---

## Environmental Justice Context

This project documents a pattern that disproportionately affects lower-income communities and communities of color:

- **Hunts Point, Bronx (10474)** — the city's largest food distribution hub processes 22 billion pounds of food annually using thousands of diesel trucks, adjacent to the Cross Bronx Expressway and Bruckner Expressway
- **Gowanus, Brooklyn (11215/11217)** — EPA Superfund site, the canal contains benzene, coal tar, and heavy metals linked to respiratory and neurological disease
- **Newtown Creek, Brooklyn/Queens (11222/11378)** — one of the most polluted waterways in the US, borders industrial truck corridors
- **Broadway Junction, Brooklyn (11233)** — active radioactive remediation site adjacent to the BQE

---

## Technologies Used

- **Python 3.12** (Miniconda)
- **pandas** — data manipulation
- **matplotlib / seaborn** — static visualizations
- **folium / branca** — interactive maps
- **requests** — API calls
- **python-dotenv** — secure API key management
- **Jupyter Notebook** — analysis environment

---

## Authors

**Jennifer Poueymirou** — Data Science Cohort
**Darnel Castor** 
The Knowledge House 
June 2026
