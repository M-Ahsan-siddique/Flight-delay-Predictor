import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FlightPreprocessor:
    def __init__(self, sample_size=200000, random_state=42):
        self.sample_size = sample_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Preprocessing artifacts to save
        self.airline_delay_rate = {}
        self.origin_delay_rate = {}
        self.dest_delay_rate = {}
        self.global_delay_rate = 0.0
        
        # New operational congestion artifacts
        self.origin_congestion = {}
        self.global_congestion = 1.0
        
        self.route_stats = {}
        self.airlines_list = []
        self.airports_list = []
        self.airport_details = {}
        
        # Fit features to scale
        self.num_cols = ['DISTANCE', 'SCHEDULED_TIME', 'FLIGHT_SPEED', 
                         'SCHEDULED_DEPARTURE_HOUR', 'SCHEDULED_ARRIVAL_HOUR',
                         'INBOUND_DELAY', 'TURNAROUND_BUFFER', 'DEPARTURE_CONGESTION']

    def load_data(self, data_dir):
        """
        Loads and joins flights, airlines, and airports datasets.
        """
        flights_path = os.path.join(data_dir, 'flights.csv')
        airlines_path = os.path.join(data_dir, 'airlines.csv')
        airports_path = os.path.join(data_dir, 'airports.csv')
        
        print("Loading datasets...")
        # Load airports and airlines first (small files)
        airlines = pd.read_csv(airlines_path)
        airports = pd.read_csv(airports_path)
        
        # Save details for the UI dashboard dropdowns
        self.airlines_list = sorted(airlines['IATA_CODE'].dropna().unique().tolist())
        self.airports_list = sorted(airports['IATA_CODE'].dropna().unique().tolist())
        self.airport_details = airports.set_index('IATA_CODE')[['AIRPORT', 'CITY', 'STATE', 'LATITUDE', 'LONGITUDE']].to_dict('index')
        
        # Determine number of rows to load or load full and sample
        use_cols = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'TAIL_NUMBER',
                    'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 
                    'SCHEDULED_TIME', 'DISTANCE', 'SCHEDULED_ARRIVAL', 'ARRIVAL_DELAY', 
                    'DEPARTURE_DELAY', 'DIVERTED', 'CANCELLED', 'WEATHER_DELAY']
        
        df = pd.read_csv(flights_path, usecols=use_cols)
        
        # Merge with origin airport coordinates for weather simulation
        df = df.merge(airports[['IATA_CODE', 'LATITUDE']], left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='left')
        df = df.rename(columns={'LATITUDE': 'ORIGIN_LAT'})
        if 'IATA_CODE' in df.columns:
            df = df.drop(columns=['IATA_CODE'])
        
        print(f"Loaded raw flights shape: {df.shape}")
        return df, airlines, airports

    def clean_data(self, df):
        """
        Cleans data: removes cancelled/diverted, filters to 3-letter IATA airport codes,
        handles missing values and outliers.
        """
        print("Cleaning data...")
        # 1. Filter out cancelled and diverted flights
        df_clean = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()
        
        # 2. Keep only standard 3-letter IATA codes for airports (remove numeric codes)
        df_clean = df_clean[
            (df_clean['ORIGIN_AIRPORT'].astype(str).str.len() == 3) &
            (df_clean['DESTINATION_AIRPORT'].astype(str).str.len() == 3)
        ]
        
        # 3. Handle outliers and check missing values in critical columns
        df_clean = df_clean[df_clean['SCHEDULED_TIME'] > 0]
        df_clean = df_clean[df_clean['DISTANCE'] > 0]
        
        # Drop rows with missing values in critical columns if any
        critical_cols = [c for c in ['ARRIVAL_DELAY', 'SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL'] if c in df_clean.columns]
        df_clean = df_clean.dropna(subset=critical_cols)
        
        print(f"Cleaned flights shape: {df_clean.shape}")
        return df_clean

    def engineer_features(self, df):
        """
        Engineers temporal, speed, weekend, seasonal, weather, and operational history features.
        """
        print("Engineering features...")
        df_feats = df.copy()
        
        # Convert date-times for chronological alignment
        dates = pd.to_datetime(df_feats[['YEAR', 'MONTH', 'DAY']])
        df_feats['SCH_DEP_DT'] = dates + pd.to_timedelta((df_feats['SCHEDULED_DEPARTURE'] // 100) * 60 + (df_feats['SCHEDULED_DEPARTURE'] % 100), unit='m')
        df_feats['SCH_ARR_DT'] = dates + pd.to_timedelta((df_feats['SCHEDULED_ARRIVAL'] // 100) * 60 + (df_feats['SCHEDULED_ARRIVAL'] % 100), unit='m')
        
        # Handle overnight arrivals
        overnight = df_feats['SCHEDULED_ARRIVAL'] < df_feats['SCHEDULED_DEPARTURE']
        df_feats.loc[overnight, 'SCH_ARR_DT'] += pd.Timedelta(days=1)
        
        # 1. Operational Flight History: Inbound delays and turnaround times
        if 'TAIL_NUMBER' not in df_feats.columns:
            df_feats['TAIL_NUMBER'] = 'UNKNOWN'
            
        df_feats = df_feats.sort_values(['TAIL_NUMBER', 'SCH_DEP_DT'])
        
        # Shifts to look at aircraft's previous flight on the same day if ARRIVAL_DELAY is present
        if 'ARRIVAL_DELAY' in df_feats.columns:
            df_feats['PREV_ARR_DELAY'] = df_feats.groupby('TAIL_NUMBER')['ARRIVAL_DELAY'].shift(1)
            df_feats['PREV_SCH_ARR_DT'] = df_feats.groupby('TAIL_NUMBER')['SCH_ARR_DT'].shift(1)
            
            # Calculate turnaround buffer (in minutes)
            time_diff = (df_feats['SCH_DEP_DT'] - df_feats['PREV_SCH_ARR_DT']).dt.total_seconds() / 60.0
            df_feats['TURNAROUND_BUFFER'] = time_diff.fillna(1440.0) # Default to 1 day (1440 mins) if no previous flight
            
            # Calculate inbound delay (only carry over if previous flight was within 12 hours)
            df_feats['INBOUND_DELAY'] = df_feats['PREV_ARR_DELAY'].fillna(0.0)
            df_feats.loc[df_feats['TURNAROUND_BUFFER'] > 720, 'INBOUND_DELAY'] = 0.0
            df_feats = df_feats.drop(columns=['PREV_ARR_DELAY', 'PREV_SCH_ARR_DT'])
        else:
            # We are running inference on custom single queries
            df_feats['TURNAROUND_BUFFER'] = 90.0
            df_feats['INBOUND_DELAY'] = 0.0
            
        # 2. Airport Congestion: Volume of departures in the same 1-hour window
        df_feats['DEP_HOUR_BLOCK'] = df_feats['SCH_DEP_DT'].dt.round('h')
        congestion = df_feats.groupby(['ORIGIN_AIRPORT', 'DEP_HOUR_BLOCK']).size().reset_index(name='DEPARTURE_CONGESTION')
        df_feats = df_feats.merge(congestion, on=['ORIGIN_AIRPORT', 'DEP_HOUR_BLOCK'], how='left')
        df_feats['DEPARTURE_CONGESTION'] = df_feats['DEPARTURE_CONGESTION'].fillna(1.0)
        
        # Clean temporary processing columns
        df_feats = df_feats.drop(columns=['SCH_DEP_DT', 'SCH_ARR_DT', 'DEP_HOUR_BLOCK'])
        
        # 3. Scheduled hours
        df_feats['SCHEDULED_DEPARTURE_HOUR'] = (df_feats['SCHEDULED_DEPARTURE'] // 100) % 24
        df_feats['SCHEDULED_ARRIVAL_HOUR'] = (df_feats['SCHEDULED_ARRIVAL'] // 100) % 24
        
        # 4. Estimated Flight Speed (miles per minute)
        df_feats['FLIGHT_SPEED'] = df_feats['DISTANCE'] / df_feats['SCHEDULED_TIME']
        
        # 5. Weekend flag
        df_feats['IS_WEEKEND'] = df_feats['DAY_OF_WEEK'].isin([6, 7]).astype(int)
        
        # 6. Season feature
        def get_season(month):
            if month in [12, 1, 2]: return 0 # Winter
            elif month in [3, 4, 5]: return 1 # Spring
            elif month in [6, 7, 8]: return 2 # Summer
            else: return 3 # Fall
        df_feats['SEASON'] = df_feats['MONTH'].apply(get_season)
        
        # 7. Weather Condition simulation (only if not already provided)
        if 'WEATHER_CONDITION' not in df_feats.columns:
            print("Simulating weather conditions...")
            weather_delay_col = df_feats['WEATHER_DELAY'].fillna(0) if 'WEATHER_DELAY' in df_feats.columns else pd.Series(0, index=df_feats.index)
            
            rng = np.random.default_rng(self.random_state)
            rand_vals = rng.random(len(df_feats))
            
            months = df_feats['MONTH'].values
            lats = df_feats['ORIGIN_LAT'].fillna(38.0).values if 'ORIGIN_LAT' in df_feats.columns else np.full(len(df_feats), 38.0)
            w_delays = weather_delay_col.values
            
            # Setup masks for vectorized conditions
            is_w_delay = w_delays > 0
            is_winter = np.isin(months, [11, 12, 1, 2])
            is_north = lats > 38.0
            is_summer = np.isin(months, [6, 7, 8, 9])
            is_south = lats < 34.0
            
            # Initialize array
            weather_conditions = np.empty(len(df_feats), dtype=object)
            
            # Weather caused delay conditions
            w_snow = is_w_delay & is_winter & is_north
            w_storm = is_w_delay & ~w_snow & (rand_vals < 0.5)
            w_wind = is_w_delay & ~w_snow & ~(rand_vals < 0.5)
            
            # Non-weather delay conditions
            no_w_delay = ~is_w_delay
            
            # Winter in North without delay
            wn_subset = no_w_delay & is_winter & is_north
            wn_snow = wn_subset & (rand_vals < 0.15)
            wn_wind = wn_subset & ~wn_snow & (rand_vals < 0.30)
            wn_rain = wn_subset & ~wn_snow & ~wn_wind & (rand_vals < 0.40)
            wn_sun = wn_subset & ~wn_snow & ~wn_wind & ~wn_rain
            
            # Summer in South without delay
            ss_subset = no_w_delay & ~wn_subset & is_summer & is_south
            ss_storm = ss_subset & (rand_vals < 0.08)
            ss_rain = ss_subset & ~ss_storm & (rand_vals < 0.25)
            ss_sun = ss_subset & ~ss_storm & ~ss_rain
            
            # Temperate default without delay
            d_subset = no_w_delay & ~wn_subset & ~ss_subset
            d_rain = d_subset & (rand_vals < 0.12)
            d_wind = d_subset & ~d_rain & (rand_vals < 0.22)
            d_sun = d_subset & ~d_rain & ~d_wind
            
            # Fill categories
            weather_conditions[w_snow] = 'Snowy'
            weather_conditions[w_storm] = 'Stormy'
            weather_conditions[w_wind] = 'Windy'
            
            weather_conditions[wn_snow] = 'Snowy'
            weather_conditions[wn_wind] = 'Windy'
            weather_conditions[wn_rain] = 'Rainy'
            weather_conditions[wn_sun] = 'Sunny'
            
            weather_conditions[ss_storm] = 'Stormy'
            weather_conditions[ss_rain] = 'Rainy'
            weather_conditions[ss_sun] = 'Sunny'
            
            weather_conditions[d_rain] = 'Rainy'
            weather_conditions[d_wind] = 'Windy'
            weather_conditions[d_sun] = 'Sunny'
            
            df_feats['WEATHER_CONDITION'] = weather_conditions
            
        # Target variable: binary delay flag (1 if arrival delay >= 15 min, else 0)
        if 'ARRIVAL_DELAY' in df_feats.columns:
            df_feats['IS_DELAYED'] = (df_feats['ARRIVAL_DELAY'] >= 15).astype(int)
        
        return df_feats

    def split_data(self, df):
        """
        Splits data chronologically:
        Train: Months 1 to 10 (Jan - Oct)
        Test: Months 11 and 12 (Nov - Dec)
        """
        print("Splitting data chronologically...")
        train_df = df[df['MONTH'] <= 10].copy()
        test_df = df[df['MONTH'] >= 11].copy()
        
        # Sample training and testing data if they are too large
        if self.sample_size and len(df) > self.sample_size:
            prop_train = len(train_df) / len(df)
            n_train_sample = int(self.sample_size * prop_train)
            n_test_sample = self.sample_size - n_train_sample
            
            train_df = train_df.sample(n=min(n_train_sample, len(train_df)), random_state=self.random_state)
            test_df = test_df.sample(n=min(n_test_sample, len(test_df)), random_state=self.random_state)
            
        print(f"Train sample shape: {train_df.shape}, Test sample shape: {test_df.shape}")
        return train_df, test_df

    def fit_encoders_and_scalers(self, train_df):
        """
        Fits Target Encoders, Congestion Maps, and StandardScaler on train set.
        """
        print("Fitting encoders and scalers on train set...")
        
        # Save route details (distance & scheduled time) for auto-lookup
        route_means = train_df.groupby(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])[['DISTANCE', 'SCHEDULED_TIME']].mean()
        self.route_stats = {
            (orig, dest): {
                'DISTANCE': float(row['DISTANCE']),
                'SCHEDULED_TIME': float(row['SCHEDULED_TIME'])
            }
            for (orig, dest), row in route_means.iterrows()
        }
        
        # 1. Compute target encoding maps (delay rates) on train data only
        self.global_delay_rate = train_df['IS_DELAYED'].mean()
        self.airline_delay_rate = train_df.groupby('AIRLINE')['IS_DELAYED'].mean().to_dict()
        self.origin_delay_rate = train_df.groupby('ORIGIN_AIRPORT')['IS_DELAYED'].mean().to_dict()
        self.dest_delay_rate = train_df.groupby('DESTINATION_AIRPORT')['IS_DELAYED'].mean().to_dict()
        
        # 2. Fit airport congestion map on train data
        self.global_congestion = train_df['DEPARTURE_CONGESTION'].mean()
        congestion_means = train_df.groupby(['ORIGIN_AIRPORT', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE_HOUR'])['DEPARTURE_CONGESTION'].mean().to_dict()
        self.origin_congestion = congestion_means
        
        # 3. Fit StandardScaler on numerical columns
        self.scaler.fit(train_df[self.num_cols])

    def transform(self, df, is_train=True):
        """
        Applies scaling, target encoding, and one-hot encoding (airlines + weather).
        """
        df_trans = df.copy()
        
        # 1. Handle missing operational feature inputs for custom testing or inference
        if 'INBOUND_DELAY' not in df_trans.columns:
            df_trans['INBOUND_DELAY'] = 0.0
        if 'TURNAROUND_BUFFER' not in df_trans.columns:
            df_trans['TURNAROUND_BUFFER'] = 90.0
            
        if not is_train:
            # Map congestion map
            df_trans['DEPARTURE_CONGESTION'] = df_trans.apply(
                lambda row: self.origin_congestion.get(
                    (row['ORIGIN_AIRPORT'], row['DAY_OF_WEEK'], row['SCHEDULED_DEPARTURE_HOUR']),
                    self.global_congestion
                ), axis=1
            ).fillna(self.global_congestion)
        
        # 2. Map target encoding delay rates
        df_trans['AIRLINE_DELAY_RATE'] = df_trans['AIRLINE'].map(self.airline_delay_rate).fillna(self.global_delay_rate)
        df_trans['ORIGIN_DELAY_RATE'] = df_trans['ORIGIN_AIRPORT'].map(self.origin_delay_rate).fillna(self.global_delay_rate)
        df_trans['DEST_DELAY_RATE'] = df_trans['DESTINATION_AIRPORT'].map(self.dest_delay_rate).fillna(self.global_delay_rate)
        
        # 3. Scale numerical columns
        df_trans[self.num_cols] = self.scaler.transform(df_trans[self.num_cols])
        
        # 4. One-hot encode Airline codes
        airlines_list = ['UA', 'AA', 'US', 'F9', 'B6', 'OO', 'AS', 'NK', 'WN', 'DL', 'EV', 'HA', 'MQ', 'VX']
        for airline in airlines_list:
            df_trans[f'AIRLINE_{airline}'] = (df_trans['AIRLINE'] == airline).astype(int)
            
        # 5. One-hot encode Weather conditions
        weather_categories = ['Sunny', 'Rainy', 'Snowy', 'Stormy', 'Windy']
        for cat in weather_categories:
            df_trans[f'WEATHER_{cat}'] = (df_trans['WEATHER_CONDITION'] == cat).astype(int)
            
        # Compile features list
        features = self.num_cols + [
            'IS_WEEKEND', 'SEASON', 
            'AIRLINE_DELAY_RATE', 'ORIGIN_DELAY_RATE', 'DEST_DELAY_RATE'
        ] + [f'AIRLINE_{airline}' for airline in airlines_list] + [f'WEATHER_{cat}' for cat in weather_categories]
        
        X = df_trans[features]
        y = df_trans['IS_DELAYED'] if 'IS_DELAYED' in df_trans.columns else None
        
        return X, y

    def run_pipeline(self, data_dir):
        """
        Executes the entire preprocessing pipeline end-to-end.
        """
        df, airlines, airports = self.load_data(data_dir)
        df_clean = self.clean_data(df)
        df_feats = self.engineer_features(df_clean)
        train_df, test_df = self.split_data(df_feats)
        
        self.fit_encoders_and_scalers(train_df)
        
        X_train, y_train = self.transform(train_df, is_train=True)
        X_test, y_test = self.transform(test_df, is_train=False)
        
        return X_train, y_train, X_test, y_test, train_df, test_df
