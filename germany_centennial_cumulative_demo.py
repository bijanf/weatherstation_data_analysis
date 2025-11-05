"""
Germany Centennial Weather Stations - DEMO with Cumulative Precipitation
=========================================================================

Modified version showing daily cumulative precipitation instead of annual totals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

print("="*90)
print("GERMANY CENTENNIAL WEATHER STATIONS - DEMO (Cumulative Precipitation)")
print("="*90 + "\n")

# Generate realistic climate data for Potsdam (1893-2024)
years = np.arange(1893, 2025)
n_years = len(years)

print(f"📊 Generating {n_years} years of synthetic climate data...")

# Temperature anomaly (same as before)
baseline_temp = 8.9
trend = np.linspace(-0.5, 1.5, n_years)
noise = np.random.normal(0, 0.6, n_years)
temp_anomaly = trend + noise

# Add some realistic extreme events
temp_anomaly[35:37] -= 1.5  # 1928-1929
temp_anomaly[54] -= 1.2     # 1947
temp_anomaly[110] += 1.5    # 2003
temp_anomaly[125] += 1.8    # 2018
temp_anomaly[126] += 1.6    # 2019
temp_anomaly[129] += 2.0    # 2022

mean_temp = baseline_temp + temp_anomaly

# Generate DAILY precipitation for each year
print("🌧️  Generating daily precipitation patterns...")

daily_data = {}
annual_totals = []

for i, year in enumerate(years):
    # Number of days in year
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    n_days = 366 if is_leap else 365

    # Generate daily precipitation (realistic pattern)
    # Base pattern: more rain in summer (June-August)
    days_of_year = np.arange(1, n_days + 1)

    # Seasonal pattern (sine wave with peak in summer)
    seasonal = 0.5 + 0.5 * np.sin((days_of_year - 80) * 2 * np.pi / 365)

    # Random daily events (exponential distribution for realistic rain)
    daily_rain = np.random.exponential(1.5, n_days) * seasonal

    # Add some extreme events randomly
    n_events = np.random.randint(3, 8)
    event_days = np.random.choice(n_days, n_events, replace=False)
    daily_rain[event_days] += np.random.exponential(15, n_events)

    # Scale to realistic annual total
    target_total = 600 + np.random.normal(0, 80)

    # Adjust for extreme years
    if year in [1943, 1978, 1982, 1997, 2018, 2023]:
        target_total -= 150  # Dry years
    elif year in [1917, 1940, 1970, 1981, 1995, 2009, 2013]:
        target_total += 150  # Wet years

    daily_rain = daily_rain * (target_total / daily_rain.sum())

    # Calculate cumulative
    cumulative = np.cumsum(daily_rain)

    daily_data[year] = {
        'days': days_of_year,
        'daily': daily_rain,
        'cumulative': cumulative,
        'total': cumulative[-1]
    }

    annual_totals.append(cumulative[-1])

# Create DataFrame for annual statistics
df = pd.DataFrame({
    'year': years,
    'temp_anomaly': temp_anomaly,
    'mean_temp': mean_temp,
    'total_prcp': annual_totals
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
print("\n🎨 Creating visualization with cumulative precipitation...")

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

# Panel 1: Temperature Anomaly (same as before)
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

textstr = f'Warming: +{warming:.2f}°C\n(pre-1950 to post-1990)'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.98, 0.05, textstr, transform=ax1.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

# Panel 2: CUMULATIVE PRECIPITATION
print("  🌧️  Plotting cumulative precipitation by day of year...")

for year in years:
    data = daily_data[year]

    # Determine color and style
    if year in extreme_years:
        color = colors[extreme_years[year]]
        alpha = 0.8
        linewidth = 2.0
        zorder = 10
    elif year > recent_threshold:
        color = colors['recent']
        alpha = 0.6
        linewidth = 1.2
        zorder = 5
    else:
        color = colors['normal']
        alpha = 0.2
        linewidth = 0.8
        zorder = 1

    ax2.plot(data['days'], data['cumulative'],
            color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)

# Add mean cumulative line
mean_cumulative = np.zeros(365)
count = np.zeros(365)
for year, data in daily_data.items():
    for i in range(min(365, len(data['cumulative']))):
        mean_cumulative[i] += data['cumulative'][i]
        count[i] += 1
mean_cumulative = mean_cumulative / count

ax2.plot(range(1, 366), mean_cumulative,
        color='black', linestyle='--', linewidth=2.5, alpha=0.8,
        label=f'Mean cumulative', zorder=15)

ax2.set_xlabel('Day of Year', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Precipitation (mm)', fontsize=12, fontweight='bold')
ax2.set_title(
    'Daily Cumulative Precipitation - Each Line is One Year',
    fontsize=13, pad=10
)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(1, 365)

# Add month labels
month_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax2.set_xticks(month_days)
ax2.set_xticklabels(month_names)

# Custom legend
legend_elements = [
    Line2D([0], [0], color=colors['hot'], linewidth=3, label='Extreme Hot Year'),
    Line2D([0], [0], color=colors['cold'], linewidth=3, label='Extreme Cold Year'),
    Line2D([0], [0], color=colors['wet'], linewidth=3, label='Extreme Wet Year'),
    Line2D([0], [0], color=colors['dry'], linewidth=3, label='Extreme Dry Year'),
    Line2D([0], [0], color=colors['recent'], linewidth=3, label='Recent Years (1995+)'),
    Line2D([0], [0], color=colors['normal'], linewidth=2, alpha=0.3, label='Normal Years'),
    Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Mean cumulative'),
]

ax2.legend(handles=legend_elements, loc='upper left',
          fontsize=10, framealpha=0.9, ncol=2)

plt.tight_layout()
save_path = "plots/germany_centennial_cumulative_DEMO.png"
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
print("\n💡 KEY OBSERVATIONS IN CUMULATIVE PLOT:")
print("   • Steep lines = wet years with heavy rainfall")
print("   • Flat/slow lines = dry years with little rain")
print("   • Line position by Dec 31 = annual total")
print("   • Can see if drought was early or late in year")
print("   • Green wet year lines rise above the pack")
print("   • Orange dry year lines stay below others")
print("\n📈 This cumulative view tells a much richer story!")
