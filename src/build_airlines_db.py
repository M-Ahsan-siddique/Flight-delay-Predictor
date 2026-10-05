import urllib.request
import csv
import io
import os
import pandas as pd

def build_global_airlines():
    url = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat'
    try:
        data = urllib.request.urlopen(url, timeout=10).read().decode('utf-8')
        reader = csv.reader(io.StringIO(data))
        of_rows = list(reader)
    except Exception as e:
        print('Error fetching OpenFlights:', e)
        of_rows = []

    airlines_dict = {}

    # Priority curated list of global airlines
    curated = [
        ('TG', 'THA', 'Thai Airways', 'Thailand'),
        ('FD', 'AIQ', 'Thai AirAsia', 'Thailand'),
        ('PG', 'BKP', 'Bangkok Airways', 'Thailand'),
        ('DD', 'NOK', 'Nok Air', 'Thailand'),
        ('SL', 'TLM', 'Thai Lion Air', 'Thailand'),
        ('VZ', 'TVJ', 'Thai Vietjet Air', 'Thailand'),
        ('PA', 'ABQ', 'Airblue', 'Pakistan'),
        ('ID', 'BTK', 'Batik Air', 'Indonesia'),
        ('OD', 'MXD', 'Batik Air Malaysia', 'Malaysia'),
        ('IU', 'SJV', 'Super Air Jet', 'Indonesia'),
        ('IW', 'WON', 'Wings Air', 'Indonesia'),
        ('PK', 'PIA', 'Pakistan International Airlines', 'Pakistan'),
        ('ER', 'SEP', 'SereneAir', 'Pakistan'),
        ('PF', 'SIF', 'AirSial', 'Pakistan'),
        ('9P', 'FJI', 'Fly Jinnah', 'Pakistan'),
        ('EK', 'UAE', 'Emirates', 'United Arab Emirates'),
        ('QR', 'QTR', 'Qatar Airways', 'Qatar'),
        ('SV', 'SVA', 'Saudia', 'Saudi Arabia'),
        ('EY', 'ETD', 'Etihad Airways', 'United Arab Emirates'),
        ('FZ', 'FDB', 'Flydubai', 'United Arab Emirates'),
        ('G9', 'ABY', 'Air Arabia', 'United Arab Emirates'),
        ('J9', 'JZR', 'Jazeera Airways', 'Kuwait'),
        ('KU', 'KAC', 'Kuwait Airways', 'Kuwait'),
        ('GF', 'GFA', 'Gulf Air', 'Bahrain'),
        ('WY', 'OMA', 'Oman Air', 'Oman'),
        ('SQ', 'SIA', 'Singapore Airlines', 'Singapore'),
        ('TR', 'TGW', 'Scoot', 'Singapore'),
        ('MH', 'MAS', 'Malaysia Airlines', 'Malaysia'),
        ('AK', 'AXM', 'AirAsia', 'Malaysia'),
        ('D7', 'XAX', 'AirAsia X', 'Malaysia'),
        ('GA', 'GIA', 'Garuda Indonesia', 'Indonesia'),
        ('JT', 'LNI', 'Lion Air', 'Indonesia'),
        ('QG', 'CTV', 'Citilink', 'Indonesia'),
        ('SJ', 'SJY', 'Sriwijaya Air', 'Indonesia'),
        ('VN', 'HVN', 'Vietnam Airlines', 'Vietnam'),
        ('VJ', 'VJC', 'VietJet Air', 'Vietnam'),
        ('QH', 'BAV', 'Bamboo Airways', 'Vietnam'),
        ('PR', 'PAL', 'Philippine Airlines', 'Philippines'),
        ('5J', 'CEB', 'Cebu Pacific', 'Philippines'),
        ('CX', 'CPA', 'Cathay Pacific', 'Hong Kong'),
        ('UO', 'HKE', 'HK Express', 'Hong Kong'),
        ('HX', 'CRK', 'Hong Kong Airlines', 'Hong Kong'),
        ('BR', 'EVA', 'EVA Air', 'Taiwan'),
        ('CI', 'CAL', 'China Airlines', 'Taiwan'),
        ('JX', 'SJX', 'STARLUX Airlines', 'Taiwan'),
        ('CA', 'CCA', 'Air China', 'China'),
        ('MU', 'CES', 'China Eastern Airlines', 'China'),
        ('CZ', 'CSN', 'China Southern Airlines', 'China'),
        ('HU', 'CHH', 'Hainan Airlines', 'China'),
        ('MF', 'CXA', 'XiamenAir', 'China'),
        ('3U', 'CSC', 'Sichuan Airlines', 'China'),
        ('ZH', 'CSZ', 'Shenzhen Airlines', 'China'),
        ('JL', 'JAL', 'Japan Airlines', 'Japan'),
        ('NH', 'ANA', 'All Nippon Airways (ANA)', 'Japan'),
        ('MM', 'APJ', 'Peach Aviation', 'Japan'),
        ('GK', 'JJP', 'Jetstar Japan', 'Japan'),
        ('KE', 'KAL', 'Korean Air', 'Republic of Korea'),
        ('OZ', 'AAR', 'Asiana Airlines', 'Republic of Korea'),
        ('7C', 'JJA', 'Jeju Air', 'Republic of Korea'),
        ('LJ', 'JNA', 'Jin Air', 'Republic of Korea'),
        ('TW', 'TWB', "T'way Air", 'Republic of Korea'),
        ('AI', 'AIC', 'Air India', 'India'),
        ('6E', 'IGO', 'IndiGo', 'India'),
        ('SG', 'SEJ', 'SpiceJet', 'India'),
        ('UK', 'VTI', 'Vistara', 'India'),
        ('IX', 'AXB', 'Air India Express', 'India'),
        ('QP', 'AKJ', 'Akasa Air', 'India'),
        ('UL', 'ALK', 'SriLankan Airlines', 'Sri Lanka'),
        ('BG', 'BBA', 'Biman Bangladesh Airlines', 'Bangladesh'),
        ('BS', 'USG', 'US-Bangla Airlines', 'Bangladesh'),
        ('KB', 'DRK', 'Drukair', 'Bhutan'),
        ('RA', 'RNA', 'Nepal Airlines', 'Nepal'),
        ('TK', 'THY', 'Turkish Airlines', 'Turkey'),
        ('PC', 'PGT', 'Pegasus Airlines', 'Turkey'),
        ('BA', 'BAW', 'British Airways', 'United Kingdom'),
        ('VS', 'VIR', 'Virgin Atlantic', 'United Kingdom'),
        ('U2', 'EZY', 'easyJet', 'United Kingdom'),
        ('LS', 'EXS', 'Jet2', 'United Kingdom'),
        ('LH', 'DLH', 'Lufthansa', 'Germany'),
        ('EW', 'EWG', 'Eurowings', 'Germany'),
        ('DE', 'CFG', 'Condor', 'Germany'),
        ('AF', 'AFR', 'Air France', 'France'),
        ('TO', 'TVF', 'Transavia France', 'France'),
        ('KL', 'KLM', 'KLM Royal Dutch Airlines', 'Netherlands'),
        ('LX', 'SWR', 'Swiss International Air Lines', 'Switzerland'),
        ('OS', 'AUA', 'Austrian Airlines', 'Austria'),
        ('SN', 'BEL', 'Brussels Airlines', 'Belgium'),
        ('IB', 'IBE', 'Iberia', 'Spain'),
        ('VY', 'VLG', 'Vueling', 'Spain'),
        ('UX', 'AEA', 'Air Europa', 'Spain'),
        ('TP', 'TAP', 'TAP Air Portugal', 'Portugal'),
        ('AZ', 'ITY', 'ITA Airways', 'Italy'),
        ('FR', 'RYR', 'Ryanair', 'Ireland'),
        ('EI', 'EIN', 'Aer Lingus', 'Ireland'),
        ('W6', 'WZZ', 'Wizz Air', 'Hungary'),
        ('LO', 'LOT', 'LOT Polish Airlines', 'Poland'),
        ('SK', 'SAS', 'Scandinavian Airlines (SAS)', 'Sweden'),
        ('DY', 'NOZ', 'Norwegian Air Shuttle', 'Norway'),
        ('AY', 'FIN', 'Finnair', 'Finland'),
        ('A3', 'AEE', 'Aegean Airlines', 'Greece'),
        ('RO', 'ROT', 'TAROM', 'Romania'),
        ('FB', 'LZB', 'Bulgaria Air', 'Bulgaria'),
        ('JU', 'ASL', 'Air Serbia', 'Serbia'),
        ('OU', 'CTN', 'Croatia Airlines', 'Croatia'),
        ('KM', 'KMM', 'Air Malta', 'Malta'),
        ('CY', 'CYP', 'Cyprus Airways', 'Cyprus'),
        ('BT', 'BTI', 'airBaltic', 'Latvia'),
        ('SU', 'AFL', 'Aeroflot', 'Russia'),
        ('S7', 'SBI', 'S7 Airlines', 'Russia'),
        ('DP', 'PBD', 'Pobeda', 'Russia'),
        ('KC', 'KZR', 'Air Astana', 'Kazakhstan'),
        ('HY', 'UZB', 'Uzbekistan Airways', 'Uzbekistan'),
        ('MS', 'MSR', 'EgyptAir', 'Egypt'),
        ('ET', 'ETH', 'Ethiopian Airlines', 'Ethiopia'),
        ('AT', 'RAM', 'Royal Air Maroc', 'Morocco'),
        ('KQ', 'KQA', 'Kenya Airways', 'Kenya'),
        ('SA', 'SAA', 'South African Airways', 'South Africa'),
        ('FA', 'SFR', 'FlySafair', 'South Africa'),
        ('WB', 'RWD', 'RwandAir', 'Rwanda'),
        ('TC', 'ATC', 'Air Tanzania', 'Tanzania'),
        ('AA', 'AAL', 'American Airlines', 'United States'),
        ('DL', 'DAL', 'Delta Air Lines', 'United States'),
        ('UA', 'UAL', 'United Airlines', 'United States'),
        ('WN', 'SWA', 'Southwest Airlines', 'United States'),
        ('B6', 'JBU', 'JetBlue Airways', 'United States'),
        ('AS', 'ASA', 'Alaska Airlines', 'United States'),
        ('NK', 'NKS', 'Spirit Airlines', 'United States'),
        ('F9', 'FFT', 'Frontier Airlines', 'United States'),
        ('HA', 'HAL', 'Hawaiian Airlines', 'United States'),
        ('OO', 'SKW', 'SkyWest Airlines', 'United States'),
        ('G4', 'AAY', 'Allegiant Air', 'United States'),
        ('SY', 'SCX', 'Sun Country Airlines', 'United States'),
        ('AC', 'ACA', 'Air Canada', 'Canada'),
        ('WS', 'WJA', 'WestJet', 'Canada'),
        ('PD', 'POE', 'Porter Airlines', 'Canada'),
        ('TS', 'TSC', 'Air Transat', 'Canada'),
        ('AM', 'AMX', 'Aeromexico', 'Mexico'),
        ('Y4', 'VOI', 'Volaris', 'Mexico'),
        ('VB', 'VIV', 'VivaAerobus', 'Mexico'),
        ('CM', 'CMP', 'Copa Airlines', 'Panama'),
        ('AV', 'AVA', 'Avianca', 'Colombia'),
        ('LA', 'LAN', 'LATAM Airlines', 'Chile'),
        ('AR', 'ARG', 'Aerolineas Argentinas', 'Argentina'),
        ('G3', 'GLO', 'Gol Transportes Aereos', 'Brazil'),
        ('AD', 'AZU', 'Azul Brazilian Airlines', 'Brazil'),
        ('QF', 'QFA', 'Qantas', 'Australia'),
        ('VA', 'VOZ', 'Virgin Australia', 'Australia'),
        ('JQ', 'JST', 'Jetstar Airways', 'Australia'),
        ('NZ', 'ANZ', 'Air New Zealand', 'New Zealand'),
        ('FJ', 'FJI', 'Fiji Airways', 'Fiji')
    ]

    for iata, icao, name, country in curated:
        label = f"{name} ({iata}) - {country}"
        airlines_dict[iata] = {
            'IATA': iata,
            'ICAO': icao,
            'Name': name,
            'Country': country,
            'Display_Label': label
        }

    # Merge OpenFlights records
    for r in of_rows:
        if len(r) >= 8:
            name = r[1].strip()
            # Clean any trailing duplicate code in name e.g. "EasyJet (DS)"
            import re
            name = re.sub(r'\s*\([A-Z0-9]{2,3}\)$', '', name).strip()
            iata = r[3].strip().upper()
            icao = r[4].strip().upper() if r[4] != '\\N' else ''
            country = r[6].strip() if r[6] != '\\N' else 'International'
            if len(iata) == 2 and iata not in ['-', '??', '\\N', 'NONE'] and name and name != '\\N':
                if iata not in airlines_dict:
                    label = f"{name} ({iata}) - {country}" if country else f"{name} ({iata})"
                    airlines_dict[iata] = {
                        'IATA': iata,
                        'ICAO': icao,
                        'Name': name,
                        'Country': country,
                        'Display_Label': label
                    }

    df_airlines = pd.DataFrame(list(airlines_dict.values()))
    df_airlines = df_airlines.sort_values(by='Display_Label')
    os.makedirs('dataset', exist_ok=True)
    csv_out = 'dataset/global_airlines.csv'
    df_airlines.to_csv(csv_out, index=False, encoding='utf-8')
    print(f"Successfully generated {csv_out} with {len(df_airlines)} airlines.")

if __name__ == '__main__':
    build_global_airlines()
