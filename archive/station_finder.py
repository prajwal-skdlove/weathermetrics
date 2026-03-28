import csv
import logging
from math import radians, cos, sin, asin, sqrt, atan2, degrees
from typing import List, Tuple, Optional
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StationFinder:
    """Find stations within specified distance and direction from a location."""
    
    # 32-point compass directions with degree ranges
    COMPASS_DIRECTIONS = {
        'N': (348.75, 11.25), 'NNE': (11.25, 33.75), 'NE': (33.75, 56.25),
        'ENE': (56.25, 78.75), 'E': (78.75, 101.25), 'ESE': (101.25, 123.75),
        'SE': (123.75, 146.25), 'SSE': (146.25, 168.75), 'S': (168.75, 191.25),
        'SSW': (191.25, 213.75), 'SW': (213.75, 236.25), 'WSW': (236.25, 258.75),
        'W': (258.75, 281.25), 'WNW': (281.25, 303.75), 'NW': (303.75, 326.25),
        'NNW': (326.25, 348.75), 'RADIUS': (0, 360)
    }
    
    EARTH_RADIUS_MILES = 3959  # Earth's radius in miles
    
    def __init__(self, csv_file: str):
        """Initialize with CSV file path."""
        self.csv_file = Path(csv_file)
        self.stations = []
        self._load_stations()
    
    def _load_stations(self) -> None:
        """Load stations from CSV file."""
        try:
            if not self.csv_file.exists():
                raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
            
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.stations.append({
                        'stid': row['stid'].strip(),
                        'latitude': float(row['latitude']),
                        'longitude': float(row['longitude'])
                    })
            logger.info(f"Loaded {len(self.stations)} stations from {self.csv_file}")
        except Exception as e:
            logger.error(f"Error loading stations: {e}")
            raise
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two points in miles using Haversine formula."""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return self.EARTH_RADIUS_MILES * c
    
    def _calculate_bearing(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate bearing from point 1 to point 2 in degrees (0-360)."""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        bearing = degrees(atan2(y, x))
        return (bearing + 360) % 360
    
    def _is_in_direction(self, bearing: float, direction: str) -> bool:
        """Check if bearing falls within specified direction."""
        if direction.upper() == 'RADIUS':
            return True
        
        if direction.upper() not in self.COMPASS_DIRECTIONS:
            logger.warning(f"Unknown direction: {direction}. Using RADIUS.")
            return True
        
        min_bearing, max_bearing = self.COMPASS_DIRECTIONS[direction.upper()]
        
        # Handle wrap-around at 0/360 degrees
        if min_bearing > max_bearing:
            return bearing >= min_bearing or bearing < max_bearing
        return min_bearing <= bearing <= max_bearing
    
    def find_stations(self, 
                     stid: Optional[str] = None,
                     latitude: Optional[float] = None,
                     longitude: Optional[float] = None,
                     distance: float = 100,
                     direction: str = 'RADIUS') -> List[dict]:
        """
        Find stations within distance and direction from a reference point.
        
        Args:
            stid: Station ID to use as reference point
            latitude: Latitude of reference point
            longitude: Longitude of reference point
            distance: Distance in miles (default 100)
            direction: Direction (default RADIUS for all directions)
        
        Returns:
            List of stations matching criteria
        """
        try:
            # Get reference point
            if stid:
                ref_station = next((s for s in self.stations if s['stid'] == stid), None)
                if not ref_station:
                    raise ValueError(f"Station {stid} not found")
                ref_lat, ref_lon = ref_station['latitude'], ref_station['longitude']
                logger.info(f"Using station {stid} as reference point")
            elif latitude is not None and longitude is not None:
                ref_lat, ref_lon = latitude, longitude
                logger.info(f"Using coordinates ({latitude}, {longitude}) as reference point")
            else:
                raise ValueError("Either stid or latitude & longitude must be provided")
            
            # Validate distance
            if distance <= 0:
                raise ValueError("Distance must be positive")
            
            # Find matching stations
            results = []
            for station in self.stations:
                dist = self._haversine_distance(
                    ref_lat, ref_lon,
                    station['latitude'], station['longitude']
                )
                
                if dist <= distance:
                    bearing = self._calculate_bearing(
                        ref_lat, ref_lon,
                        station['latitude'], station['longitude']
                    )
                    
                    if self._is_in_direction(bearing, direction):
                        results.append({
                            'stid': station['stid'],
                            'latitude': station['latitude'],
                            'longitude': station['longitude'],
                            'distance': round(dist, 2),
                            'bearing': round(bearing, 2)
                        })
            
            results.sort(key=lambda x: x['distance'])
            logger.info(f"Found {len(results)} stations")
            return results
        
        except Exception as e:
            logger.error(f"Error finding stations: {e}")
            raise
    
    def save_results(self, results: List[dict], output_file: str) -> None:
        """Save results to CSV file."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not results:
                logger.warning("No results to save")
                return
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Find weather stations within specified distance and direction'
    )
    parser.add_argument('csv_file', help='Path to stations.csv file')
    parser.add_argument('-s', '--stid', help='Station ID as reference point')
    parser.add_argument('-lat', '--latitude', type=float, help='Latitude of reference point')
    parser.add_argument('-lon', '--longitude', type=float, help='Longitude of reference point')
    parser.add_argument('-d', '--distance', type=float, default=100, help='Distance in miles (default: 100)')
    parser.add_argument('-dir', '--direction', default='RADIUS', help='Direction: N, NE, E, SE, S, SW, W, NW, or RADIUS (default: RADIUS)')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file path')
    
    args = parser.parse_args()
    
    try:
        finder = StationFinder(args.csv_file)
        results = finder.find_stations(
            stid=args.stid,
            latitude=args.latitude,
            longitude=args.longitude,
            distance=args.distance,
            direction=args.direction
        )
        # finder.save_results(results, args.output)
        print(results)        
    except Exception as e:
        logger.error(f"Application error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
    

    # Find stations 100 miles of a specific station
    # python station_finder.py ../stations.csv -s 76903099999 -d 100 -dir RADIUS -o results.csv
    # Find stations 50 miles in all directions from coordinates
    # python station_finder.py stations.csv -lat 35.5 -lon -97.5 -d 50 -o results.csv
    # Find stations 200 miles northeast using radius search
    # python station_finder.py ../stations.csv -s 76903099999 -d 100 -dir NE -o results.csv    