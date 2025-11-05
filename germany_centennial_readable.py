"""
Germany Centennial Weather Stations - READABLE LABELS VERSION
==============================================================

Clean visualization with 2018 drought and improved label readability.
Labels are strategically positioned to avoid overlap.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Set random seed
np.random.seed(42)

print("="*90)
print("GERMANY CENTENNIAL - READABLE LABELS VERSION")
print("="*90 + "\n")

# Generate data (same as before)
years = np.arange(1893, 2025)
n_years = len(years)

print(f"📊 Generating {n_years} years of data...")

baseline_temp = 8.9
trend = np.linspace(-0.5, 1.5, n_years)
noise = np.random.normal(0, 0.6, n_years)
temp_anomaly = trend + noise

temp_anomaly[35:37] -= 1.5
temp_anomaly[54] -= 1.2
temp_anomaly[110] += 1.5
temp_anomaly[125] += 1.8
temp_anomaly[126] += 1.6
temp_anomaly[129] += 2.0

mean_temp = baseline_temp + temp_anomaly

daily_data = {}
annual_totals = []

for i, year in enumerate(years):
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    n_days = 366 if is_leap else 365

    days_of_year = np.arange(1, n_days + 1)
    seasonal = 0.5 + 0.5 * np.sin((days_of_year - 80) * 2 * np.pi / 365)
    daily_rain = np.random.exponential(1.5, n_days) * seasonal

    n_events = np.random.randint(3, 8)
    event_days = np.random.choice(n_days, n_events, replace=False)
    daily_rain[event_days] += np.random.exponential(15, n_events)

    target_total = 600 + np.random.normal(0, 80)

    if year == 2018:
        target_total = 350
    elif year in [1943, 1976, 1959, 2003]:
        target_total -= 150
    elif year in [1917, 1940, 1970, 2002]:
        target_total += 150

    daily_rain = daily_rain * (target_total / daily_rain.sum())
    cumulative = np.cumsum(daily_rain)

    daily_data[year] = {
        'days': days_of_year,
        'cumulative': cumulative,
        'total': cumulative[-1]
    }
    annual_totals.append(cumulative[-1])

df = pd.DataFrame({
    'year': years,
    'temp_anomaly': temp_anomaly,
    'mean_temp': mean_temp,
    'total_prcp': annual_totals
})

# Get 5 driest (ensuring 2018)
driest_years = df.nsmallest(5, 'total_prcp')[['year', 'total_prcp']].copy()
driest_years = driest_years.sort_values('total_prcp')
driest_years_list = driest_years['year'].tolist()

print(f"\n🏜️  5 Driest Years:")
for _, row in driest_years.iterrows():
    print(f"   {int(row['year'])}: {row['total_prcp']:.0f} mm")

recent_threshold = 1994

# Colors
colors = {
    'dry': '#ff7f0e',
    'dry_2018': '#d62728',
    'recent': '#9467bd',
    'normal': '#7f7f7f'
}

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

station_name = "Potsdam Säkularstation (Demo)"
fig.suptitle(
    f'Century-Long Climate Analysis: {station_name}\n'
    f'{int(df["year"].min())}–{int(df["year"].max())} ({len(df)} years)',
    fontsize=16, fontweight='bold', y=0.995
)

# Panel 1: Temperature
for _, row in df.iterrows():
    year = int(row['year'])
    anomaly = row['temp_anomaly']

    if year > recent_threshold:
        color, alpha, lw = colors['recent'], 0.7, 1.5
    else:
        color, alpha, lw = colors['normal'], 0.4, 1

    ax1.plot([year, year], [0, anomaly], color=color, alpha=alpha, linewidth=lw)

z = np.polyfit(df['year'], df['temp_anomaly'], 1)
p = np.poly1d(z)
ax1.plot(df['year'], p(df['year']),
        'k--', linewidth=2.5, alpha=0.8,
        label=f'Trend: +{z[0]*100:.2f}°C/century')

early = df[df['year'] < 1950]
late = df[df['year'] >= 1990]
warming = late['mean_temp'].mean() - early['mean_temp'].mean()

ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax1.set_xlabel('Year', fontsize=13, fontweight='bold')
ax1.set_ylabel('Temperature Anomaly (°C)\nvs 1961-1990', fontsize=13, fontweight='bold')
ax1.set_title('Annual Mean Temperature Anomaly', fontsize=14, pad=10, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper left', fontsize=12, framealpha=0.9)

textstr = f'Warming:\n+{warming:.2f}°C'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, linewidth=2)
ax1.text(0.98, 0.05, textstr, transform=ax1.transAxes, fontsize=13,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=props, fontweight='bold')

# Panel 2: Precipitation with READABLE LABELS
print("\n🎨 Creating plot with readable labels...")

# Background: normal years
for year in years:
    if year not in driest_years_list and year <= recent_threshold:
        data = daily_data[year]
        ax2.plot(data['days'], data['cumulative'],
                color=colors['normal'], alpha=0.12, linewidth=0.8, zorder=1)

# Recent years
for year in years:
    if year > recent_threshold and year not in driest_years_list:
        data = daily_data[year]
        ax2.plot(data['days'], data['cumulative'],
                color=colors['recent'], alpha=0.4, linewidth=1.2, zorder=5)

# Driest years
driest_data = []
for idx, (_, row) in enumerate(driest_years.iterrows()):
    year = int(row['year'])
    data = daily_data[year]

    if year == 2018:
        color = colors['dry_2018']
        lw = 4.0
        alpha = 1.0
        zorder = 20
    else:
        color = colors['dry']
        lw = 2.8
        alpha = 0.95
        zorder = 15

    line, = ax2.plot(data['days'], data['cumulative'],
                    color=color, alpha=alpha, linewidth=lw, zorder=zorder)

    driest_data.append({
        'year': year,
        'final_value': data['cumulative'][-1],
        'color': color,
        'is_2018': year == 2018
    })

# IMPROVED LABEL POSITIONING - spread vertically to avoid overlap
print("  🏷️  Positioning labels to avoid overlap...")

# Sort by final value to position labels
driest_data_sorted = sorted(driest_data, key=lambda x: x['final_value'])

# Calculate positions to spread labels vertically
y_positions = []
min_spacing = 50  # minimum mm spacing between labels

for i, item in enumerate(driest_data_sorted):
    if i == 0:
        y_pos = item['final_value']
    else:
        # Ensure minimum spacing from previous label
        y_pos = max(item['final_value'], y_positions[-1] + min_spacing)
    y_positions.append(y_pos)

# Add labels with better positioning
for item, y_pos in zip(driest_data_sorted, y_positions):
    year = item['year']
    final_value = item['final_value']
    color = item['color']

    if item['is_2018']:
        label_text = f"2018\n{final_value:.0f} mm\n⚠️ EXTREME"
        fontsize = 13
        fontweight = 'bold'
        box_lw = 3
        pad = 0.5
    else:
        label_text = f"{year}\n{final_value:.0f} mm"
        fontsize = 12
        fontweight = 'bold'
        box_lw = 2
        pad = 0.4

    # Draw connection line from data to label
    ax2.plot([365, 375], [final_value, y_pos],
            color=color, linewidth=1.5, alpha=0.6, zorder=10)

    # Add label
    ax2.text(380, y_pos, label_text,
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
            verticalalignment='center',
            bbox=dict(boxstyle=f'round,pad={pad}',
                     facecolor='white',
                     edgecolor=color,
                     linewidth=box_lw,
                     alpha=0.95),
            zorder=25)

# Mean line
mean_cumulative = np.zeros(365)
count = np.zeros(365)
for year, data in daily_data.items():
    for i in range(min(365, len(data['cumulative']))):
        mean_cumulative[i] += data['cumulative'][i]
        count[i] += 1
mean_cumulative = mean_cumulative / count

ax2.plot(range(1, 366), mean_cumulative,
        color='black', linestyle='--', linewidth=3, alpha=0.8,
        label='Mean', zorder=12)

ax2.set_xlabel('Month', fontsize=13, fontweight='bold')
ax2.set_ylabel('Cumulative Precipitation (mm)', fontsize=13, fontweight='bold')
ax2.set_title('Daily Cumulative Precipitation - Driest Years Labeled',
             fontsize=14, pad=10, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(1, 450)  # Extended for labels
ax2.set_ylim(0, 850)

# Month labels
month_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax2.set_xticks(month_days)
ax2.set_xticklabels(month_names, fontsize=11)

# Legend
legend_elements = [
    Line2D([0], [0], color=colors['dry_2018'], linewidth=4,
           label='2018 Exceptional Drought'),
    Line2D([0], [0], color=colors['dry'], linewidth=3,
           label='Other Driest Years'),
    Line2D([0], [0], color=colors['recent'], linewidth=2,
           label='Recent Years (1995+)'),
    Line2D([0], [0], color=colors['normal'], linewidth=2, alpha=0.3,
           label='Normal Years'),
    Line2D([0], [0], color='black', linestyle='--', linewidth=2.5,
           label='Long-term Mean'),
]

ax2.legend(handles=legend_elements, loc='upper left',
          fontsize=12, framealpha=0.95, edgecolor='black')

plt.tight_layout()
save_path = "plots/germany_centennial_READABLE.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Plot saved to: {save_path}")

print(f"\n{'='*90}")
print(f"READABLE LABELS - IMPROVEMENTS:")
print(f"{'='*90}")
print(f"✓ Labels positioned to avoid overlap")
print(f"✓ Connection lines from data to labels")
print(f"✓ Larger fonts (12-13pt)")
print(f"✓ Bold text for better visibility")
print(f"✓ Thicker label boxes")
print(f"✓ Extended plot area for label space")
print(f"✓ 2018 prominently featured in RED")
print(f"{'='*90}\n")

print("✅ Labels are now READABLE!")
