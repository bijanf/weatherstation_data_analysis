"""
Germany Centennial Weather Stations - CLEAN VERSION (2018 Drought Featured)
=============================================================================

Simplified visualization highlighting 2018 drought plus 4 other driest years.
2018 was one of the most severe droughts in German history.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

print("="*90)
print("GERMANY CENTENNIAL WEATHER STATIONS - 2018 DROUGHT FEATURED")
print("="*90 + "\n")

# Generate realistic climate data for Potsdam (1893-2024)
years = np.arange(1893, 2025)
n_years = len(years)

print(f"📊 Generating {n_years} years of synthetic climate data...")

# Temperature anomaly
baseline_temp = 8.9
trend = np.linspace(-0.5, 1.5, n_years)
noise = np.random.normal(0, 0.6, n_years)
temp_anomaly = trend + noise

# Add realistic extreme events
temp_anomaly[35:37] -= 1.5  # 1928-1929
temp_anomaly[54] -= 1.2     # 1947
temp_anomaly[110] += 1.5    # 2003
temp_anomaly[125] += 1.8    # 2018 - hot AND dry!
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

    # Generate daily precipitation
    days_of_year = np.arange(1, n_days + 1)
    seasonal = 0.5 + 0.5 * np.sin((days_of_year - 80) * 2 * np.pi / 365)
    daily_rain = np.random.exponential(1.5, n_days) * seasonal

    # Add extreme events
    n_events = np.random.randint(3, 8)
    event_days = np.random.choice(n_days, n_events, replace=False)
    daily_rain[event_days] += np.random.exponential(15, n_events)

    # Scale to realistic annual total
    target_total = 600 + np.random.normal(0, 80)

    # Adjust for extreme years - make 2018 VERY dry
    if year == 2018:
        target_total = 350  # EXTREME DROUGHT like reality
    elif year in [1943, 1976, 1959, 2003]:
        target_total -= 150  # Other notable dry years
    elif year in [1917, 1940, 1970, 2002]:
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

# Create DataFrame
df = pd.DataFrame({
    'year': years,
    'temp_anomaly': temp_anomaly,
    'mean_temp': mean_temp,
    'total_prcp': annual_totals
})

print(f"✅ Generated data for {df['year'].min()}-{df['year'].max()}")

# Get the 5 driest years - ensuring 2018 is included
driest_candidates = df.nsmallest(10, 'total_prcp').copy()

# Make sure 2018 is in the list
if 2018 not in driest_candidates['year'].values:
    # Add 2018 manually
    year_2018 = df[df['year'] == 2018].copy()
    driest_candidates = pd.concat([driest_candidates.head(4), year_2018])

# Take top 5, ensuring 2018 is included
if 2018 in driest_candidates['year'].values:
    driest_years = driest_candidates.nsmallest(5, 'total_prcp')[['year', 'total_prcp']].copy()
else:
    # Manually ensure 2018 is included
    driest_years = pd.concat([
        driest_candidates.nsmallest(4, 'total_prcp'),
        df[df['year'] == 2018]
    ])[['year', 'total_prcp']]

driest_years = driest_years.sort_values('total_prcp')
driest_years_list = driest_years['year'].tolist()

print(f"\n🏜️  5 Driest Years (including 2018 exceptional drought):")
for _, row in driest_years.iterrows():
    year_label = int(row['year'])
    marker = " ⚠️  EXTREME" if year_label == 2018 else ""
    print(f"   {year_label}: {row['total_prcp']:.0f} mm{marker}")

recent_threshold = df['year'].max() - 30

# Create visualization
print("\n🎨 Creating visualization with 2018 drought highlighted...")

colors = {
    'dry': '#ff7f0e',      # Orange for driest years
    'dry_2018': '#d62728', # Special red for 2018
    'recent': '#9467bd',   # Purple for recent years
    'normal': '#7f7f7f'    # Grey for everything else
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

station_name = "Potsdam Säkularstation (Demo Data)"
fig.suptitle(
    f'Century-Long Climate Analysis: {station_name}\n'
    f'{int(df["year"].min())}–{int(df["year"].max())} ({len(df)} years)',
    fontsize=16, fontweight='bold', y=0.995
)

# Panel 1: Temperature Anomaly
print("  📈 Plotting temperature anomalies...")
for _, row in df.iterrows():
    year = int(row['year'])
    anomaly = row['temp_anomaly']

    if year > recent_threshold:
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
print("  🌧️  Plotting cumulative precipitation with 2018 featured...")

# Plot all normal years in grey (background)
for year in years:
    if year not in driest_years_list and year <= recent_threshold:
        data = daily_data[year]
        ax2.plot(data['days'], data['cumulative'],
                color=colors['normal'], alpha=0.15, linewidth=0.8, zorder=1)

# Plot recent years in purple (except 2018)
for year in years:
    if year > recent_threshold and year not in driest_years_list:
        data = daily_data[year]
        ax2.plot(data['days'], data['cumulative'],
                color=colors['recent'], alpha=0.5, linewidth=1.2, zorder=5)

# Plot the other 4 driest years in orange
driest_lines = []
for year in driest_years_list:
    if year != 2018:
        data = daily_data[year]
        line, = ax2.plot(data['days'], data['cumulative'],
                        color=colors['dry'], alpha=0.9, linewidth=2.5, zorder=10)
        driest_lines.append((year, line, data, 'dry'))

# Plot 2018 in special red color with thicker line
if 2018 in driest_years_list:
    data_2018 = daily_data[2018]
    line_2018, = ax2.plot(data_2018['days'], data_2018['cumulative'],
                          color=colors['dry_2018'], alpha=1.0, linewidth=3.5, zorder=15)
    driest_lines.append((2018, line_2018, data_2018, 'extreme'))

# Add labels for the driest years
print("  🏷️  Adding labels for driest years...")
for year, line, data, drought_type in driest_lines:
    final_value = data['cumulative'][-1]

    if drought_type == 'extreme':
        # Special formatting for 2018
        label_text = f'2018 ⚠️\n{final_value:.0f}mm\nEXTREME'
        bbox_color = colors['dry_2018']
        fontsize = 11
        fontweight = 'bold'
    else:
        label_text = f'{int(year)}\n{final_value:.0f}mm'
        bbox_color = colors['dry']
        fontsize = 10
        fontweight = 'normal'

    ax2.annotate(label_text,
                xy=(365, final_value),
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=fontsize,
                fontweight=fontweight,
                color=bbox_color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor=bbox_color, linewidth=2, alpha=0.9),
                zorder=20)

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
    'Daily Cumulative Precipitation - 2018 Exceptional Drought Highlighted',
    fontsize=13, pad=10
)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(1, 390)  # Extended for labels

# Add month labels
month_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax2.set_xticks(month_days)
ax2.set_xticklabels(month_names)

# Simplified legend
legend_elements = [
    Line2D([0], [0], color=colors['dry_2018'], linewidth=4, label='2018 Exceptional Drought'),
    Line2D([0], [0], color=colors['dry'], linewidth=3, label='Other Driest Years'),
    Line2D([0], [0], color=colors['recent'], linewidth=2, label='Recent Years (1995+)'),
    Line2D([0], [0], color=colors['normal'], linewidth=2, alpha=0.3, label='Normal Years'),
    Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Mean cumulative'),
]

ax2.legend(handles=legend_elements, loc='upper left',
          fontsize=11, framealpha=0.9)

plt.tight_layout()
save_path = "plots/germany_centennial_2018.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Plot saved to: {save_path}")

# Print summary
print(f"\n{'='*90}")
print(f"2018 EXCEPTIONAL DROUGHT FEATURED")
print(f"{'='*90}")
print(f"\n🎨 Color Scheme:")
print(f"   🔴 Red (thick): 2018 exceptional drought - hot AND dry year")
print(f"   🟠 Orange: Other driest years")
print(f"   🟣 Purple: Recent years (1995+)")
print(f"   ⚫ Grey: Normal years")
print(f"\n🏜️  5 Driest Years:")
for _, row in driest_years.iterrows():
    year_label = int(row['year'])
    marker = " ⚠️  EXCEPTIONAL - Hot & Dry" if year_label == 2018 else ""
    print(f"   {year_label}: {row['total_prcp']:.0f} mm{marker}")
print(f"\n🌡️  Temperature: +{warming:.2f}°C warming (pre-1950 to post-1990)")
print(f"🔥 2018 was both VERY HOT (+{temp_anomaly[125]:.1f}°C) and VERY DRY!")
print(f"{'='*90}\n")

print("✅ 2018 drought properly featured!")
print("\n💡 2018 STANDS OUT:")
print("   • Red line at bottom = severe drought")
print("   • Also one of hottest years (purple bar high in top panel)")
print("   • Combination of heat + drought = exceptional year")
print("   • Historically significant for German climate")
print("\n📈 Perfect for highlighting the 2018 drought event!")
