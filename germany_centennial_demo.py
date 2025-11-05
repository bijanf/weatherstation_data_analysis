"""
Germany Centennial Weather Stations - DEMO VERSION
===================================================

Creates a demonstration plot with realistic synthetic data showing what
the actual visualization will look like with real data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Set random seed for reproducibility
np.random.seed(42)

print("="*90)
print("GERMANY CENTENNIAL WEATHER STATIONS - DEMO")
print("Creating demonstration with realistic synthetic data")
print("="*90 + "\n")

# Generate realistic climate data for Potsdam (1893-2024)
years = np.arange(1893, 2025)
n_years = len(years)

print(f"📊 Generating {n_years} years of synthetic climate data...")

# Temperature anomaly with realistic warming trend
# Start around -0.5°C, end around +1.5°C (realistic for Germany)
baseline_temp = 8.9  # Potsdam mean temp
trend = np.linspace(-0.5, 1.5, n_years)
noise = np.random.normal(0, 0.6, n_years)  # Natural variability
temp_anomaly = trend + noise

# Add some realistic extreme events
# 1928-1929 cold winters
temp_anomaly[35:37] -= 1.5
# 1947 harsh winter
temp_anomaly[54] -= 1.2
# 2003, 2018, 2019, 2022 hot years
temp_anomaly[110] += 1.5  # 2003
temp_anomaly[125] += 1.8  # 2018
temp_anomaly[126] += 1.6  # 2019
temp_anomaly[129] += 2.0  # 2022

mean_temp = baseline_temp + temp_anomaly

# Precipitation with realistic variability
mean_prcp = 600  # mm/year for Potsdam
prcp_noise = np.random.normal(0, 80, n_years)
total_prcp = mean_prcp + prcp_noise

# Add extreme dry/wet years
total_prcp[50] -= 150  # 1943 drought
total_prcp[85] -= 120  # 1978 drought
total_prcp[120] += 150  # 2013 wet
total_prcp[125] -= 180  # 2018 extreme drought

# Ensure no negative precipitation
total_prcp = np.maximum(total_prcp, 300)

# Create DataFrame
df = pd.DataFrame({
    'year': years,
    'temp_anomaly': temp_anomaly,
    'mean_temp': mean_temp,
    'total_prcp': total_prcp
})

print(f"✅ Generated data for {df['year'].min()}-{df['year'].max()}")

# Identify extreme years
def identify_extremes(df):
    extremes = {}
    threshold = 1.5

    temp_std = df['temp_anomaly'].std()
    temp_mean = df['temp_anomaly'].mean()
    prcp_std = df['total_prcp'].std()
    prcp_mean = df['total_prcp'].mean()

    for _, row in df.iterrows():
        year = int(row['year'])
        temp_z = (row['temp_anomaly'] - temp_mean) / temp_std
        prcp_z = (row['total_prcp'] - prcp_mean) / prcp_std

        if temp_z > threshold:
            extremes[year] = 'hot'
        elif temp_z < -threshold:
            extremes[year] = 'cold'
        elif prcp_z > threshold:
            extremes[year] = 'wet'
        elif prcp_z < -threshold:
            extremes[year] = 'dry'

    return extremes

extreme_years = identify_extremes(df)
print(f"🔥 Identified {len(extreme_years)} extreme years")

# Create visualization
print("\n🎨 Creating visualization...")

colors = {
    'hot': '#d62728',
    'cold': '#1f77b4',
    'wet': '#2ca02c',
    'dry': '#ff7f0e',
    'normal': '#7f7f7f',
    'recent': '#9467bd'
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

station_name = "Potsdam Säkularstation (Demo Data)"
fig.suptitle(
    f'Century-Long Climate Analysis: {station_name}\n'
    f'{int(df["year"].min())}–{int(df["year"].max())} ({len(df)} years)',
    fontsize=16, fontweight='bold', y=0.995
)

recent_threshold = df['year'].max() - 30

# Panel 1: Temperature Anomaly
print("  📈 Plotting temperature anomalies...")
for _, row in df.iterrows():
    year = int(row['year'])
    anomaly = row['temp_anomaly']

    if year in extreme_years:
        color = colors[extreme_years[year]]
        alpha = 0.8
        linewidth = 2
    elif year > recent_threshold:
        color = colors['recent']
        alpha = 0.7
        linewidth = 1.5
    else:
        color = colors['normal']
        alpha = 0.4
        linewidth = 1

    ax1.plot([year, year], [0, anomaly], color=color, alpha=alpha, linewidth=linewidth)

# Trend line
z = np.polyfit(df['year'], df['temp_anomaly'], 1)
p = np.poly1d(z)
ax1.plot(df['year'], p(df['year']),
        'k--', linewidth=2.5, alpha=0.8,
        label=f'Linear trend: +{z[0]*100:.2f}°C per century')

# Calculate warming
early = df[df['year'] < 1950]
late = df[df['year'] >= 1990]
warming = late['mean_temp'].mean() - early['mean_temp'].mean()

ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('Temperature Anomaly (°C)\nrelative to 1961-1990',
              fontsize=12, fontweight='bold')
ax1.set_title(
    'Annual Mean Temperature Anomaly - Warming Trend Visible in Recent Decades',
    fontsize=13, pad=10
)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)

# Text box with warming stats
textstr = f'Warming: +{warming:.2f}°C\n(pre-1950 to post-1990)'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.98, 0.05, textstr, transform=ax1.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

# Panel 2: Precipitation
print("  🌧️  Plotting precipitation totals...")
for _, row in df.iterrows():
    year = int(row['year'])
    prcp = row['total_prcp']

    if year in extreme_years:
        color = colors[extreme_years[year]]
        alpha = 0.8
        linewidth = 2
    elif year > recent_threshold:
        color = colors['recent']
        alpha = 0.7
        linewidth = 1.5
    else:
        color = colors['normal']
        alpha = 0.4
        linewidth = 1

    ax2.plot([year, year], [0, prcp], color=color, alpha=alpha, linewidth=linewidth)

# Mean line
mean_prcp = df['total_prcp'].mean()
ax2.axhline(y=mean_prcp, color='black', linestyle='--',
           linewidth=2, alpha=0.7, label=f'Mean: {mean_prcp:.0f} mm')

ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
ax2.set_ylabel('Annual Precipitation (mm)', fontsize=12, fontweight='bold')
ax2.set_title(
    'Annual Total Precipitation - Extreme Dry and Wet Years Highlighted',
    fontsize=13, pad=10
)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper left', fontsize=11, framealpha=0.9)

# Custom legend
legend_elements = [
    Line2D([0], [0], color=colors['hot'], linewidth=3, label='Extreme Hot Year'),
    Line2D([0], [0], color=colors['cold'], linewidth=3, label='Extreme Cold Year'),
    Line2D([0], [0], color=colors['wet'], linewidth=3, label='Extreme Wet Year'),
    Line2D([0], [0], color=colors['dry'], linewidth=3, label='Extreme Dry Year'),
    Line2D([0], [0], color=colors['recent'], linewidth=3, label='Recent Years (1995+)'),
    Line2D([0], [0], color=colors['normal'], linewidth=2, label='Normal Years'),
]

ax2.legend(handles=legend_elements, loc='upper right',
          fontsize=10, framealpha=0.9, ncol=2)

plt.tight_layout()
save_path = "plots/germany_centennial_DEMO.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Plot saved to: {save_path}")

# Print summary
print(f"\n{'='*90}")
print(f"CLIMATE SUMMARY: {station_name}")
print(f"{'='*90}")
print(f"\n📊 Data Coverage: {int(df['year'].min())}–{int(df['year'].max())} ({len(df)} years)")
print(f"\n🌡️  Temperature:")
print(f"  Mean annual: {df['mean_temp'].mean():.2f}°C")
print(f"  Warmest year: {int(df.loc[df['mean_temp'].idxmax(), 'year'])} ({df['mean_temp'].max():.2f}°C)")
print(f"  Coldest year: {int(df.loc[df['mean_temp'].idxmin(), 'year'])} ({df['mean_temp'].min():.2f}°C)")
print(f"\n🔥 Warming Trend:")
print(f"  Pre-1950 mean: {early['mean_temp'].mean():.2f}°C")
print(f"  Post-1990 mean: {late['mean_temp'].mean():.2f}°C")
print(f"  Total warming: +{warming:.2f}°C")
print(f"  Trend rate: +{z[0]*100:.2f}°C per century")
print(f"\n🌧️  Precipitation:")
print(f"  Mean annual: {df['total_prcp'].mean():.0f} mm")
print(f"  Wettest year: {int(df.loc[df['total_prcp'].idxmax(), 'year'])} ({df['total_prcp'].max():.0f} mm)")
print(f"  Driest year: {int(df.loc[df['total_prcp'].idxmin(), 'year'])} ({df['total_prcp'].min():.0f} mm)")
print(f"\n⚠️  Extreme Years ({len(extreme_years)} total):")
for ext_type in ['hot', 'cold', 'wet', 'dry']:
    years_list = [y for y, t in extreme_years.items() if t == ext_type]
    if years_list:
        print(f"  {ext_type.capitalize()}: {', '.join(map(str, sorted(years_list)))}")
print(f"{'='*90}\n")

print("✅ DEMO complete!")
print("\n💡 KEY OBSERVATIONS:")
print("   • Recent years (purple) cluster in the upper half - showing warming")
print("   • Grey normal years dominate the historical record")
print("   • Extreme hot years (red) mostly in recent decades")
print("   • Extreme cold years (blue) mostly in early 1900s")
print("   • The warming trend is clearly visible and public-friendly!")
print("\n📈 This is what your plot will look like with REAL DWD data!")
