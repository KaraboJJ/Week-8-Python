# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta

# Set visualization style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Display all outputs in notebook
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Data Collection and Loading
def load_covid_data():
    """Load COVID-19 data from Our World in Data"""
    try:
        # Try to load from online source
        url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        df = pd.read_csv(url)
        print("✅ Successfully loaded data from Our World in Data")

        # Save local copy
        df.to_csv("owid-covid-data.csv", index=False)
        return df

    except Exception as e:
        print(f"⚠️ Online load failed: {e}\nTrying local file...")
        try:
            df = pd.read_csv("owid-covid-data.csv")
            print("✅ Successfully loaded local data file")
            return df
        except:
            print("❌ Could not load data file")
            return None

# Load the data
print("🔍 Loading COVID-19 dataset...")
covid_df = load_covid_data()

if covid_df is None:
    raise SystemExit("Failed to load COVID-19 data. Please check your data source.")

#Data Exploration
# Markdown cell (create this in Jupyter by changing cell type to Markdown)
"""
## 2. Data Exploration

Let's examine the structure and contents of our dataset
"""

def explore_data(df):
    """Perform initial data exploration"""
    from IPython.display import display

    print("\n📊 Dataset Overview:")
    print(f"Shape: {df.shape} (rows, columns)")

    print("\n🔎 First 5 rows:")
    display(df.head())

    print("\n📋 Data columns:")
    display(pd.DataFrame(df.dtypes, columns=['Data Type']))

    print("\n🔍 Missing values summary:")
    missing_values = df.isnull().sum().sort_values(ascending=False)
    display(missing_values[missing_values > 0].to_frame('Missing Values'))

    print("\n📅 Date range:")
    print(f"Start: {df['date'].min()}  |  End: {df['date'].max()}")

    print("\n🌍 Locations (first 20):")
    display(pd.DataFrame(df['location'].unique()[:20], columns=['Countries']))

print("\n🔬 Exploring the dataset...")
explore_data(covid_df)

# Data Cleaning
# Markdown cell
"""
## 3. Data Cleaning

Preparing our data for analysis by:
- Filtering selected countries
- Handling missing values
- Calculating important metrics
"""

def clean_covid_data(df, countries=None):
    """Clean and prepare COVID-19 data for analysis"""
    from IPython.display import display

    print("\n🧹 Cleaning data...")

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Filter countries if specified
    if countries:
        if isinstance(countries, str):
            countries = [countries]
        df = df[df['location'].isin(countries)].copy()
        print(f"\n🌐 Filtered data for {len(countries)} countries:")
        display(pd.DataFrame(countries, columns=['Selected Countries']))

    # Select key columns
    key_columns = [
        'date', 'location', 'continent', 'population',
        'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
        'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
        'icu_patients', 'hosp_patients', 'reproduction_rate'
    ]
    df = df[key_columns]

    # Forward fill missing values within each country
    df_clean = df.groupby('location').apply(lambda x: x.ffill())

    # Fill remaining NA with 0 for case/death/vaccination metrics
    fill_cols = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths',
                'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']
    df_clean[fill_cols] = df_clean[fill_cols].fillna(0)

    # Calculate derived metrics
    df_clean['death_rate'] = df_clean['total_deaths'] / df_clean['total_cases']
    df_clean['vaccination_rate'] = df_clean['people_vaccinated'] / df_clean['population']
    df_clean['cases_per_million'] = (df_clean['total_cases'] / df_clean['population']) * 1e6

    print("\n✅ Data cleaning complete")
    print(f"Final shape: {df_clean.shape}")

    return df_clean

# Clean data for selected countries
selected_countries = ['United States', 'India', 'Brazil', 'Germany', 'Kenya', 'South Africa']
print(f"\n🔄 Processing data for: {', '.join(selected_countries)}")
clean_df = clean_covid_data(covid_df, selected_countries)

# Show cleaned data sample
print("\n🧼 Cleaned data sample:")
clean_df.head()

# Exploratory Data Analysis (EDA)
# Markdown cell
""" 4. Exploratory Data Analysis Visualizing trends and patterns in the COVID-19 data"""

def perform_eda(df):
    """Perform exploratory data analysis"""
    from IPython.display import display

    print("\n🔍 Beginning EDA...")

    # Get latest data
    latest_date = df['date'].max()
    df_latest = df[df['date'] == latest_date]

    # Summary statistics
    print("\n📈 Summary statistics:")
    display(df.describe())

    # Latest cases and deaths
    print(f"\n📅 Latest data as of {latest_date.date()}:")
    latest_stats = df_latest[['location', 'total_cases', 'total_deaths',
                             'people_vaccinated', 'death_rate']]
    display(latest_stats.sort_values('total_cases', ascending=False))

    # Time series plots
    print("\n📈 Total Cases Over Time (Log Scale)")
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df, x='date', y='total_cases', hue='location')
    plt.title('COVID-19 Total Cases Over Time', pad=20)
    plt.ylabel('Total Cases (log scale)')
    plt.xlabel('Date')
    plt.yscale('log')
    plt.show()

    # New cases comparison
    print("\n📉 Daily New Cases (7-day moving average)")
    plt.figure(figsize=(14, 8))
    for country in df['location'].unique():
        country_data = df[df['location'] == country]
        plt.plot(country_data['date'], country_data['new_cases'].rolling(7).mean(),
                label=country, alpha=0.8)
    plt.title('Daily New Cases Trend', pad=20)
    plt.ylabel('New Cases')
    plt.xlabel('Date')
    plt.legend()
    plt.show()

    # Death rate comparison
    print("\n💀 Case Fatality Rate Comparison")
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_latest, x='location', y='death_rate')
    plt.title('Case Fatality Rate by Country', pad=20)
    plt.ylabel('Death Rate (deaths/cases)')
    plt.xlabel('Country')
    plt.xticks(rotation=45)

    # Add value labels
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2%}",
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10),
                   textcoords='offset points')
    plt.show()

print("\n📊 Performing exploratory analysis...")
perform_eda(clean_df)

# Vaccination Analysis
# Markdown cell
"""
## 5. Vaccination Analysis

Tracking vaccination progress across countries
"""

def analyze_vaccinations(df):
    """Analyze vaccination progress"""
    print("\n💉 Analyzing vaccination data...")

    # Vaccination progress over time
    print("\n📈 Total Vaccinations Over Time")
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df, x='date', y='people_vaccinated', hue='location')
    plt.title('Total People Vaccinated Over Time', pad=20)
    plt.ylabel('People Vaccinated')
    plt.xlabel('Date')
    plt.show()

    # Vaccination rate comparison
    print("\n📊 Vaccination Rate Over Time")
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df, x='date', y='vaccination_rate', hue='location')
    plt.title('Vaccination Rate Trend', pad=20)
    plt.ylabel('Vaccination Rate (proportion of population)')
    plt.xlabel('Date')
    plt.show()

    # Latest vaccination status
    latest_date = df['date'].max()
    df_latest = df[df['date'] == latest_date]

    print(f"\n🛡️ Vaccination Rates as of {latest_date.date()}")
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_latest, x='location', y='vaccination_rate')
    plt.title('Vaccination Rate by Country', pad=20)
    plt.ylabel('Vaccination Rate')
    plt.xlabel('Country')
    plt.xticks(rotation=45)

    # Add percentage labels
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}",
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10),
                   textcoords='offset points')
    plt.show()

analyze_vaccinations(clean_df)

# Choropleth Map Visualization
# Markdown cell
"""
## 6. Geographic Distribution

World map visualization of COVID-19 cases
"""

def create_choropleth(df):
    """Create a world map visualization"""
    try:
        print("\n🌍 Creating interactive world map...")

        # Prepare data for world map
        latest_date = df['date'].max()
        world_df = df[df['date'] == latest_date]

        fig = px.choropleth(world_df,
                            locations="location",
                            locationmode='country names',
                            color="cases_per_million",
                            hover_name="location",
                            hover_data=["total_cases", "total_deaths", "people_vaccinated"],
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title=f"COVID-19 Cases per Million (as of {latest_date.date()})",
                            height=600)
        fig.show()
    except Exception as e:
        print(f"❌ Could not create choropleth map: {e}")
        print("Tip: Make sure plotly is installed (pip install plotly)")

create_choropleth(clean_df)

# Insights and Reporting
# Markdown cell
"""
## 7. Key Insights

Summary of important findings from our analysis
"""

def generate_insights(df):
    """Generate key insights from the analysis"""
    from IPython.display import Markdown

    # Get latest data
    latest_date = df['date'].max()
    df_latest = df[df['date'] == latest_date]

    insights = [
        "## 🔍 COVID-19 Analysis Insights\n",
        f"*Data as of {latest_date.date()}*"
    ]

    # 1. Country with highest cases
    max_cases = df_latest.loc[df_latest['total_cases'].idxmax()]
    insights.append(f"\n1. **{max_cases['location']}** has the highest total cases: {max_cases['total_cases']:,.0f}")

    # 2. Country with highest death rate
    filtered = df_latest[df_latest['total_cases'] > 10000]  # Filter for meaningful rates
    max_death_rate = filtered.loc[filtered['death_rate'].idxmax()]
    insights.append(f"\n2. **{max_death_rate['location']}** has the highest death rate: {max_death_rate['death_rate']:.2%}")

    # 3. Vaccination leader
    max_vaccination = df_latest.loc[df_latest['vaccination_rate'].idxmax()]
    insights.append(f"\n3. **{max_vaccination['location']}** has the highest vaccination rate: {max_vaccination['vaccination_rate']:.1%}")

    # 4. Case trends analysis
    insights.append("\n4. **Case Trend Analysis:**")
    for country in df['location'].unique():
        country_data = df[df['location'] == country]
        peak_cases = country_data['new_cases'].max()
        peak_date = country_data.loc[country_data['new_cases'].idxmax(), 'date']
        insights.append(f"   - {country} peaked at {peak_cases:,.0f} daily cases on {peak_date.date()}")

    # 5. Vaccination progress comparison
    insights.append("\n5. **Vaccination Progress Comparison:**")
    for country in df['location'].unique():
        country_data = df[df['location'] == country]
        vaccinated = country_data['people_vaccinated'].iloc[-1]
        population = country_data['population'].iloc[-1]
        insights.append(f"   - {country} has vaccinated {vaccinated/population:.1%} of its population")

    # Display as Markdown
    display(Markdown('\n'.join(insights)))

print("\n💡 Generating key insights...")
generate_insights(clean_df)

# Interactive Analysis
# Markdown cell
"""
## 8. Interactive Analysis (Optional)

Explore the data interactively
"""

def interactive_analysis(df):
    """Allow interactive country and metric selection"""
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output

        print("\n🔄 Setting up interactive controls...")

        # Create widgets
        country_selector = widgets.SelectMultiple(
            options=df['location'].unique(),
            value=['United States', 'India'],
            description='Countries:',
            layout={'width': '400px'}
        )

        metric_selector = widgets.Dropdown(
            options=['total_cases', 'total_deaths', 'people_vaccinated', 'new_cases'],
            value='total_cases',
            description='Metric:'
        )

        date_range = widgets.DateRangeSlider(
            value=(df['date'].min(), df['date'].max()),
            min=df['date'].min(),
            max=df['date'].max(),
            description='Date Range:',
            layout={'width': '500px'}
        )

        # Output area
        out = widgets.Output()

        # Interactive plot function
        def update_plot(change):
            with out:
                clear_output(wait=True)
                countries = country_selector.value
                metric = metric_selector.value
                start_date, end_date = date_range.value

                mask = (df['location'].isin(countries)) & \
                       (df['date'] >= pd.to_datetime(start_date)) & \
                       (df['date'] <= pd.to_datetime(end_date))

                plt.figure(figsize=(14, 7))
                sns.lineplot(data=df[mask], x='date', y=metric, hue='location')
                plt.title(f'COVID-19 {metric.replace("_", " ").title()} Over Time', pad=20)
                plt.ylabel(metric.replace("_", " ").title())
                plt.xlabel('Date')
                if metric in ['total_cases', 'total_deaths', 'people_vaccinated']:
                    plt.yscale('log')
                plt.show()

        # Set up observers
        country_selector.observe(update_plot, names='value')
        metric_selector.observe(update_plot, names='value')
        date_range.observe(update_plot, names='value')

        # Initial display
        display(widgets.VBox([
            widgets.HBox([country_selector, metric_selector]),
            date_range,
            out
        ]))
        update_plot(None)

    except ImportError:
        print("❌ ipywidgets not available. Install with: pip install ipywidgets")
        print("Then restart your Jupyter notebook and enable widgets with:")
        print("jupyter nbextension enable --py widgetsnbextension")

# Uncomment to enable interactive analysis
# interactive_analysis(clean_df)
