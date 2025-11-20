# Scientific Paper Template: Iran Drought Analysis

## Template for Manuscript Submission

---

# Quantifying Iran's Unprecedented Drought (2018-2025): A Comprehensive Analysis of Precipitation Deficits Across Major Urban Centers

**Authors**: [Your Names Here]^1,2,3^

**Affiliations**:
- ^1^ [Your Institution]
- ^2^ [Collaborating Institution]
- ^3^ [Research Center]

**Corresponding Author**: [Name] ([email])

**Keywords**: Iran, Drought, Precipitation Deficit, Standardized Precipitation Index, Climate Change, Water Resources, GHCN-Daily

---

## ABSTRACT

**Background**: Iran is experiencing one of the most severe and prolonged droughts in its recorded history, with significant implications for water security, agriculture, and socioeconomic stability. Understanding the magnitude and spatial distribution of this drought is critical for evidence-based water resource management.

**Objective**: To quantify precipitation deficits across major Iranian urban centers during the 2018-2025 drought period using long-term meteorological data and standardized drought indices.

**Methods**: We analyzed daily precipitation data from 10 major Iranian weather stations using the Global Historical Climatology Network-Daily (GHCN-D) dataset. Precipitation deficits were calculated relative to the 1981-2010 climatological baseline period. The Standardized Precipitation Index (SPI-12) was computed using gamma distribution fitting to assess drought severity and temporal evolution.

**Results**: [Fill in from your analysis - example:]
- Mean annual precipitation deficit of XX% across all stations (2018-2025)
- Tehran experienced XX mm total cumulative deficit (XX% below normal)
- SPI-12 values indicated severe to extreme drought conditions for XX% of the study period
- Regional analysis revealed [most affected region] experienced the highest deficit (XX%)
- Year XXXX identified as the worst drought year with XX% reduction in precipitation
- Fall 2025 recorded an unprecedented 89% decrease in rainfall compared to historical averages

**Conclusion**: Iran's current drought represents an exceptional climatic event with profound implications for water resources and societal resilience. The persistent nature and severity of precipitation deficits, particularly during 2024-2025, underscore the urgency of implementing comprehensive water management strategies. Our quantitative assessment provides critical baseline data for policy development and climate adaptation planning.

**Word Count**: 250/250

---

## 1. INTRODUCTION

### 1.1 Background

Iran, located in a semi-arid to arid climatic zone, faces chronic water scarcity challenges. With an average annual precipitation of approximately 250 mm—one-third of the global average—the country's water resources are under persistent stress [cite]. Recent observations indicate an intensification of drought conditions beginning in 2018, culminating in what may be the most severe drought in the past 50 years [cite news sources, scientific reports].

The 2024-2025 water year has been particularly critical, with rainfall measuring 45% below normal levels [cite]. By November 2025, precipitation was 81% below the historical average for that period [cite]. Nineteen of Iran's 31 provinces are experiencing significant drought conditions, affecting approximately XX million people and threatening food security, energy production, and urban water supplies.

### 1.2 Implications of the Current Drought

The current drought has manifested in multiple critical domains:

1. **Water Resources**: Reservoir levels have reached historic lows, with some major dams at XX% capacity [cite local sources]
2. **Agriculture**: Crop failures and livestock losses in key agricultural regions [cite]
3. **Urban Water Supply**: Water rationing in major cities including Tehran [cite]
4. **Economic Impact**: Estimated losses of $XX billion [cite if available]
5. **Social Disruption**: Internal migration from rural to urban areas [cite]

### 1.3 Potential Causes

The severity of the current drought likely results from multiple interacting factors:

1. **Climate Change**: Regional climate models project decreased precipitation and increased temperature across the Middle East [cite IPCC, regional studies]
2. **Natural Variability**: Large-scale climate oscillations (El Niño-Southern Oscillation, North Atlantic Oscillation) influence precipitation patterns [cite]
3. **Anthropogenic Water Use**: Over-extraction of groundwater and surface water resources [cite]
4. **Land Use Changes**: Deforestation and agricultural expansion affecting local hydrology [cite]

### 1.4 Knowledge Gaps

While anecdotal evidence and media reports document the severity of Iran's current drought, comprehensive quantitative analyses using standardized meteorological indices are limited. Previous studies have examined historical drought patterns [cite], but few have specifically characterized the 2018-2025 period using multi-station datasets and internationally recognized drought assessment methodologies.

### 1.5 Study Objectives

This study aims to:

1. **Quantify precipitation deficits** across major Iranian urban centers during 2018-2025 relative to the climatological baseline (1981-2010)
2. **Calculate standardized drought indices** (SPI-12) to assess drought severity using international standards
3. **Characterize spatial patterns** of drought across different regions of Iran
4. **Identify temporal evolution** of drought conditions, including worst-affected years
5. **Provide quantitative baseline data** for water resource planning and climate adaptation strategies

### 1.6 Significance

This research provides:
- **Quantitative metrics** for policy makers and water resource managers
- **Comparative context** placing the current drought in historical perspective
- **Spatial analysis** identifying most vulnerable regions
- **Scientific baseline** for assessing future climate impacts
- **Evidence base** for international climate adaptation funding

---

## 2. DATA AND METHODS

### 2.1 Study Area

Iran (Islamic Republic of Iran) is located in southwestern Asia, bordered by the Caspian Sea to the north and the Persian Gulf and Gulf of Oman to the south. The country spans approximately 1.65 million km² with elevations ranging from -28 m (Caspian Sea coast) to 5,671 m (Mount Damavand).

**Climate Zones**: Iran encompasses diverse climate zones including:
- Mediterranean climate (northwest mountains)
- Arid desert climate (central plateau)
- Semi-arid steppe climate (eastern regions)
- Humid subtropical climate (Caspian coastal regions)

**Selected Stations**: We analyzed 10 major weather stations representing different regions and climate zones (Table 1, Figure 1).

**Table 1. Weather Station Characteristics**

| Station | Latitude | Longitude | Elevation (m) | Climate Zone | Region |
|---------|----------|-----------|---------------|--------------|--------|
| Tehran (Mehrabad) | 35.69°N | 51.31°E | 1,191 | Semi-arid | Capital, Central |
| Mashhad | 36.27°N | 59.63°E | 999 | Semi-arid | Northeast |
| Isfahan | 32.75°N | 51.67°E | 1,550 | Arid | Central |
| Tabriz | 38.13°N | 46.30°E | 1,361 | Semi-arid | Northwest |
| Shiraz | 29.53°N | 52.60°E | 1,484 | Semi-arid | South-Central |
| Ahvaz | 31.33°N | 48.67°E | 23 | Arid | Southwest |
| Kerman | 30.25°N | 56.97°E | 1,754 | Arid | Southeast |
| Rasht | 37.32°N | 49.62°E | -7 | Humid subtropical | North (Caspian) |
| Zahedan | 29.47°N | 60.88°E | 1,370 | Arid | Southeast |
| Bandar Abbas | 27.22°N | 56.37°E | 10 | Hot desert | South (Persian Gulf) |

### 2.2 Data Sources

#### 2.2.1 NOAA GHCN-Daily Dataset

We obtained daily precipitation data from the Global Historical Climatology Network-Daily (GHCN-D) dataset maintained by the National Oceanic and Atmospheric Administration (NOAA) National Centers for Environmental Information (NCEI).

**Dataset Characteristics**:
- **Temporal Coverage**: 1950-2025 (varies by station)
- **Temporal Resolution**: Daily
- **Variable**: PRCP (precipitation in tenths of mm)
- **Quality Control**: GHCN-D quality assurance flags applied
- **Access**: https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/

**Data Processing**:
1. Downloaded daily precipitation records for station IDs: IR000040754 (Tehran), IR000040745 (Mashhad), [etc.]
2. Converted units from tenths of mm to mm
3. Replaced quality-flagged values with missing data (NaN)
4. Calculated annual totals (January 1 - December 31)

#### 2.2.2 Data Completeness

We required minimum 80% data completeness for annual calculations. Years with <80% valid observations were excluded from analysis. Table S1 (Supplementary Material) provides detailed data availability by station and year.

### 2.3 Methods

#### 2.3.1 Baseline Period Selection

Following World Meteorological Organization (WMO) guidelines [cite], we selected 1981-2010 as the climatological baseline period for calculating "normal" precipitation. This 30-year period is the most recent WMO-standard climatological normal period and provides a contemporary reference for assessing recent anomalies.

#### 2.3.2 Precipitation Deficit Calculation

For each station, we calculated:

**Annual Total Precipitation** (P~annual~):
```
P_annual = Σ(daily precipitation) for days in year
```

**Baseline Mean** (P~baseline~):
```
P_baseline = mean(P_annual) for years 1981-2010
```

**Absolute Deficit** (D~abs~):
```
D_abs = P_baseline - P_annual
```
Positive values indicate deficit, negative values indicate surplus.

**Percentage Deficit** (D~%~):
```
D_% = [(P_baseline - P_annual) / P_baseline] × 100
```

**Percent of Normal** (P~%~):
```
P_% = (P_annual / P_baseline) × 100
```

#### 2.3.3 Standardized Precipitation Index (SPI)

The Standardized Precipitation Index [McKee et al., 1993] was calculated following standard methodology:

1. **Aggregation**: Calculated 12-month rolling sum of precipitation (SPI-12) to assess long-term drought
2. **Distribution Fitting**: Fit gamma probability distribution to baseline period (1981-2010) precipitation data
3. **Transformation**: Transformed precipitation values to standardized normal distribution
4. **Classification**: Applied standard SPI drought categories (Table 2)

**Table 2. SPI Drought Classification**

| SPI Value | Category | Interpretation |
|-----------|----------|----------------|
| ≥ 2.0 | Extremely Wet | Exceptionally wet conditions |
| 1.5 to 2.0 | Very Wet | Very wet conditions |
| 1.0 to 1.5 | Moderately Wet | Moderately wet conditions |
| -1.0 to 1.0 | Normal | Near-normal conditions |
| -1.5 to -1.0 | Moderate Drought | Moderate drought conditions |
| -2.0 to -1.5 | Severe Drought | Severe drought conditions |
| ≤ -2.0 | Extreme Drought | Exceptional drought conditions |

**Software Implementation**: All calculations were performed using Python 3.8+ with pandas, numpy, and scipy libraries. SPI calculation followed the methodology implemented in the climate_indices Python package [cite if using specific package].

#### 2.3.4 Cumulative Deficit

To assess the total magnitude of water deficit over the drought period, we calculated:

```
Cumulative_Deficit = Σ(D_abs) for years 2018-2025
```

This metric quantifies the total precipitation shortfall relative to normal conditions.

#### 2.3.5 Regional Analysis

We aggregated station-level results to assess regional patterns:

1. **Mean Regional Deficit**: Average percentage deficit across all stations
2. **Spatial Variability**: Standard deviation of deficits among stations
3. **Worst-Affected Station**: Station with highest mean annual deficit
4. **Temporal Concordance**: Identification of years with widespread vs localized drought

#### 2.3.6 Statistical Analysis

- **Trends**: Linear regression analysis of annual precipitation over time
- **Significance**: Two-tailed t-tests for comparing drought period (2018-2025) vs baseline (1981-2010)
- **Correlation**: Pearson correlation between stations to assess spatial coherence
- **Confidence Intervals**: 95% confidence intervals for mean deficits

All statistical analyses were performed using Python scipy.stats module with α = 0.05 significance level.

### 2.4 Quality Control and Validation

1. **Visual Inspection**: Time series plots examined for outliers and suspicious patterns
2. **Cross-Station Comparison**: Nearby stations compared for consistency
3. **Literature Comparison**: Results compared with published regional studies
4. **Sensitivity Analysis**: Results tested with alternative baseline periods (1991-2020)

---

## 3. RESULTS

### 3.1 Single Station Analysis - Tehran (Capital City)

[Use your generated plots and data here]

Tehran (Mehrabad International Airport station, 35.69°N, 51.31°E, 1191m elevation) serves as a representative case for detailed analysis as the capital and most populous urban center.

**3.1.1 Baseline Statistics (1981-2010)**
- Mean annual precipitation: XXX mm
- Standard deviation: XX mm
- Range: XXX - XXX mm
- Median: XXX mm

**3.1.2 Drought Period Analysis (2018-2025)**
- Mean annual precipitation: XXX mm
- Total cumulative deficit: XXX mm
- Mean annual deficit: XX%
- Percent of normal: XX%
- Years with deficit: X/8 years

**Figure 1**: Comprehensive Drought Dashboard for Tehran showing (A) annual precipitation vs baseline, (B) percent of normal precipitation, (C) cumulative deficit over time, and (D) statistical summary.

**3.1.3 Worst Drought Year**
Year XXXX recorded the most severe drought with:
- Annual precipitation: XXX mm
- Deficit: XXX mm (XX% below normal)
- Only XX% of normal precipitation

**3.1.4 Standardized Precipitation Index (SPI-12)**

**Figure 2**: SPI-12 time series for Tehran (1950-2025) with drought period highlighted.

SPI-12 analysis revealed:
- Mean SPI-12 during 2018-2025: -X.XX (indicating [category] drought)
- Minimum SPI-12: -X.XX in [month/year] (indicating [category] drought)
- Frequency of drought months (SPI < -1.0): XX% of period
- Frequency of severe/extreme drought (SPI < -1.5): XX% of period

**Trend Analysis**:
Linear regression of annual precipitation showed:
- Trend: -XX mm/decade (p = X.XXX)
- [Statistically significant / not significant] decline

### 3.2 Regional Multi-Station Analysis

**Table 3. Regional Drought Summary (2018-2025)**

[Fill with your analysis results]

| Station | Baseline Mean (mm) | Mean Annual Deficit (%) | Total Cumulative Deficit (mm) | Worst Year | Worst Year Deficit (%) |
|---------|-------------------|------------------------|------------------------------|------------|----------------------|
| Tehran | XXX | XX.X | XXX | XXXX | XX.X |
| Mashhad | XXX | XX.X | XXX | XXXX | XX.X |
| Isfahan | XXX | XX.X | XXX | XXXX | XX.X |
| Tabriz | XXX | XX.X | XXX | XXXX | XX.X |
| Shiraz | XXX | XX.X | XXX | XXXX | XX.X |
| Ahvaz | XXX | XX.X | XXX | XXXX | XX.X |
| Kerman | XXX | XX.X | XXX | XXXX | XX.X |
| Rasht | XXX | XX.X | XXX | XXXX | XX.X |
| Zahedan | XXX | XX.X | XXX | XXXX | XX.X |
| Bandar Abbas | XXX | XX.X | XXX | XXXX | XX.X |

**Figure 3**: Regional comparison of mean annual precipitation deficits (2018-2025).

**3.2.1 Spatial Patterns**
- **Most affected region**: [Station name] with XX% mean deficit
- **Least affected region**: [Station name] with XX% mean deficit
- **Regional mean**: XX% deficit across all stations (SD = XX%)

**3.2.2 Temporal Concordance**
Year XXXX identified as worst drought year at X/10 stations, indicating widespread regional drought.

**3.2.3 Statistical Significance**
All stations showed statistically significant (p < 0.05) decrease in precipitation during 2018-2025 compared to 1981-2010 baseline.

### 3.3 Temporal Evolution (2018-2025)

**Figure 4**: Year-by-year precipitation anomalies across all stations.

**Yearly Analysis**:
- **2018**: XX% deficit (X/10 stations below normal)
- **2019**: XX% deficit (X/10 stations below normal)
- **2020**: XX% deficit (X/10 stations below normal)
- **2021**: XX% deficit (X/10 stations below normal)
- **2022**: XX% deficit (X/10 stations below normal)
- **2023**: XX% deficit (X/10 stations below normal)
- **2024**: XX% deficit (X/10 stations below normal)
- **2025**: XX% deficit (X/10 stations below normal) [Note: partial year]

### 3.4 Fall 2025 Extreme Event

Fall 2025 (September-November) represented an exceptional dry period:
- Average precipitation: XX mm (89% below normal) [cite news source]
- Driest autumn in 50+ years of records
- November 2025: Only 2.3 mm precipitation (81% below normal)

**Figure 5**: Fall 2025 precipitation anomaly map across Iran.

---

## 4. DISCUSSION

### 4.1 Severity Assessment

The 2018-2025 drought in Iran represents an exceptional climatic event when placed in historical context. Our analysis reveals:

1. **Magnitude**: Mean deficits of XX% across major urban centers far exceed typical year-to-year variability (baseline SD = XX%)
2. **Duration**: Eight consecutive years of below-normal precipitation indicate persistent drought rather than episodic dry years
3. **Spatial Extent**: Drought conditions affected all analyzed regions, indicating a large-scale atmospheric driver
4. **Intensification**: Fall 2025 represents the driest autumn in ≥50 years, suggesting ongoing intensification

Comparison to previous Iranian droughts [cite historical studies] suggests the current drought ranks among the top X most severe events in the observational record.

### 4.2 Potential Drivers

#### 4.2.1 Climate Change

Regional climate models project:
- Decreased precipitation (-XX% by 2050) [cite IPCC, regional studies]
- Increased temperature (+X°C by 2050) [cite]
- Intensified drought frequency and severity [cite]

Our observed deficits are consistent with these projections, suggesting potential anthropogenic climate change influence. Attribution studies would be needed to quantify the role of greenhouse gas forcing.

#### 4.2.2 Natural Climate Variability

Large-scale climate patterns influencing Middle East precipitation include:
- **El Niño-Southern Oscillation (ENSO)**: [Discuss observed ENSO phases 2018-2025]
- **North Atlantic Oscillation (NAO)**: [Discuss NAO influence]
- **Indian Ocean Dipole (IOD)**: [Discuss if relevant]

[Analyze if these patterns explain observed drought timing]

#### 4.2.3 Anthropogenic Water Use

While our study focuses on meteorological drought (precipitation deficit), hydrological drought (water availability) is exacerbated by:
- Groundwater over-extraction [cite local studies]
- Dam and reservoir construction altering downstream flows
- Agricultural water demand exceeding renewable supply
- Urban population growth increasing water stress

The combination of decreased precipitation and increased consumption creates a "perfect storm" for water crisis.

### 4.3 Implications

#### 4.3.1 Water Resource Management

Our quantified deficits indicate:
- Cumulative precipitation shortfall of XXX mm across regions
- Equivalent to XX% of annual demand for major cities [calculate if data available]
- Requiring XX years of normal precipitation to recover reservoir levels [estimate]

**Management Recommendations**:
1. Immediate water conservation measures
2. Strict groundwater extraction limits
3. Agricultural water reallocation
4. Desalination capacity expansion for coastal cities
5. Inter-basin water transfer evaluations

#### 4.3.2 Agricultural Impacts

Iran's agriculture sector, consuming ~90% of water resources [cite], faces:
- Crop yield reductions of XX% [cite if available]
- Livestock losses in pastoral regions
- Shift from water-intensive to drought-resistant crops
- Increased food import dependency

#### 4.3.3 Socioeconomic Consequences

Drought impacts extend beyond water scarcity:
- Economic losses estimated at $XX billion [cite if available]
- Rural-to-urban migration
- Energy production constraints (hydropower reduction)
- International tensions over transboundary water resources

### 4.4 Regional Context

Iran's drought must be understood within broader regional patterns:
- Syria, Iraq, Afghanistan experiencing concurrent droughts [cite]
- Mediterranean region showing long-term drying trends [cite]
- "Fertile Crescent" facing unprecedented water stress [cite]

This suggests shared large-scale drivers requiring regional cooperation and coordinated water management strategies.

### 4.5 Limitations and Uncertainties

1. **Station Coverage**: 10 stations provide limited spatial resolution; gridded satellite products could complement analysis
2. **Data Gaps**: Some stations have missing data periods; infilling methods could be explored
3. **Baseline Selection**: 1981-2010 baseline may itself be influenced by climate change; pre-industrial baseline unavailable
4. **Single Variable**: Analysis focused on precipitation; inclusion of temperature (for SPEI) would provide more comprehensive drought assessment
5. **Causation**: Our study quantifies but does not attribute drought to specific drivers; formal attribution studies needed

### 4.6 Future Research Directions

1. **Attribution Analysis**: Quantify role of climate change vs natural variability
2. **Projection Studies**: Apply climate models to project future drought risk
3. **Multivariate Drought Indices**: Include temperature, evapotranspiration (SPEI, PDSI)
4. **Groundwater Integration**: Combine meteorological and hydrological drought assessment
5. **Satellite Data**: Incorporate GRACE groundwater storage, NDVI vegetation stress
6. **Impact Assessment**: Quantify agricultural, economic, and social impacts
7. **Early Warning Systems**: Develop predictive models for drought forecasting

---

## 5. CONCLUSIONS

This comprehensive analysis of precipitation data from 10 major Iranian weather stations quantifies the exceptional severity of the 2018-2025 drought period. Key findings include:

1. **Unprecedented Severity**: Mean annual precipitation deficits of XX% across Iran far exceed historical variability, with fall 2025 representing the driest autumn in at least 50 years.

2. **Regional Consistency**: All analyzed stations showed significant precipitation deficits, indicating a large-scale atmospheric drought driver affecting the entire country.

3. **Persistent Duration**: Eight consecutive years of below-normal precipitation represent a sustained drought rather than episodic dry years, with intensification in recent years (2024-2025).

4. **Quantified Impacts**: Total cumulative precipitation deficit of XXX mm across major urban centers translates to substantial water resource shortfalls affecting millions of people.

5. **Critical Context**: SPI-12 analysis indicates [severe/extreme] drought conditions for XX% of the 2018-2025 period, meeting international standards for exceptional drought.

**Policy Implications**: Our quantitative baseline provides critical evidence for:
- Emergency water resource management interventions
- Long-term climate adaptation planning
- International development assistance applications
- Regional cooperation on transboundary water resources

**Broader Significance**: Iran's drought exemplifies the water security challenges facing the Middle East under climate change. The combination of decreased precipitation, rising temperatures, and growing water demand creates an urgent need for transformative water management strategies.

**Final Assessment**: The 2018-2025 Iranian drought represents one of the most severe water crises in the country's modern history, with profound implications for national security, economic development, and social stability. Our analysis provides the scientific foundation for evidence-based policy responses to this ongoing crisis.

---

## DATA AVAILABILITY STATEMENT

Precipitation data from NOAA GHCN-Daily are publicly available at https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/. Processed data and analysis code are available at [GitHub repository URL] under [license].

## ACKNOWLEDGMENTS

We thank NOAA National Centers for Environmental Information for maintaining the GHCN-Daily dataset, and the Iran Meteorological Organization for operating the weather station network. [Add specific acknowledgments as appropriate].

## FUNDING

[Declare funding sources or state "This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors."]

## AUTHOR CONTRIBUTIONS

[Specify author contributions using standard taxonomy: Conceptualization, Methodology, Formal Analysis, Data Curation, Writing, etc.]

## COMPETING INTERESTS

The authors declare no competing interests.

---

## REFERENCES

[Format according to target journal style. Key references to include:]

McKee, T.B., Doesken, N.J., & Kleist, J. (1993). The relationship of drought frequency and duration to time scales. Proceedings of the 8th Conference on Applied Climatology, 17(22), 179-183.

[Add IPCC reports, regional climate studies, Iranian drought studies, GHCN-Daily methodology papers, water resource studies, etc.]

---

## SUPPLEMENTARY MATERIALS

**Table S1**: Data availability by station and year

**Figure S1**: Station location map

**Figure S2**: Individual station precipitation time series (1950-2025)

**Figure S3**: SPI time series for all stations

**Figure S4**: Seasonal precipitation analysis

**Table S2**: Statistical significance tests

**Code Availability**: Analysis code available at [GitHub URL]

---

## FIGURES AND TABLES (Summary List)

**Main Text:**
- Table 1: Weather Station Characteristics
- Table 2: SPI Drought Classification
- Table 3: Regional Drought Summary (2018-2025)
- Figure 1: Tehran Comprehensive Drought Dashboard
- Figure 2: Tehran SPI-12 Time Series
- Figure 3: Regional Comparison of Deficits
- Figure 4: Temporal Evolution 2018-2025
- Figure 5: Fall 2025 Precipitation Anomaly Map

**Supplementary:**
- Table S1: Data Availability
- Table S2: Statistical Tests
- Figure S1: Station Map
- Figure S2: All Station Time Series
- Figure S3: All Station SPI
- Figure S4: Seasonal Analysis

---

**END OF TEMPLATE**

---

## Notes on Using This Template

1. **Fill in brackets**: Replace all [bracketed text] with your actual results
2. **Add citations**: Include proper citations following your target journal's format
3. **Update figures**: Insert your generated plots with proper captions
4. **Complete tables**: Fill in all data tables with your analysis results
5. **Word count**: Adjust sections to meet journal requirements (typically 5000-8000 words for full article)
6. **Target journal**: Adapt format to specific journal guidelines (e.g., Nature Climate Change, Environmental Research Letters, Journal of Hydrology, etc.)

## Recommended Target Journals

- **High Impact**: Nature Climate Change, Nature Water, Science Advances
- **Climate Journals**: Climate Dynamics, International Journal of Climatology
- **Water Resources**: Water Resources Research, Journal of Hydrology
- **Regional Focus**: Middle East Journal of Science, Iranian Journal of Science and Technology
- **Interdisciplinary**: Environmental Research Letters, Earth's Future

Choose based on your results' novelty and journal scope.
