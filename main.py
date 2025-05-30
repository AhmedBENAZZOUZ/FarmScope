
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union
import json
import requests
import time
import os
import re
import math
from datetime import datetime, timedelta
import ee
import geemap
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
port = int(os.getenv("PORT", 8000))

app = FastAPI()

# Enable CORS with more specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Update the LocationRequest model to handle complex context objects
class MessageContext(BaseModel):
    role: str
    content: str

class LocationRequest(BaseModel):
    location: Optional[str] = None
    coordinates: Optional[List[float]] = None
    message: str
    context: Optional[List[Union[str, Dict[str, Any], MessageContext]]] = []
    isInitialRequest: Optional[bool] = False
    existingData: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    message: str
    ndvi: Optional[float] = None
    ndwi: Optional[float] = None
    soil_moisture: Optional[float] = None
    weather: Optional[Dict[str, Any]] = None
    map_url: Optional[str] = None
    advice: Optional[str] = None

# Initialize Earth Engine with error handling
try:
    GEE_CREDENTIALS = os.getenv('GEE_CREDENTIALS')
    if not GEE_CREDENTIALS:
        raise ValueError("GEE credentials not found in environment variables")
    
    credentials_dict = json.loads(GEE_CREDENTIALS)
    credentials = ee.ServiceAccountCredentials(
        email=credentials_dict['client_email'],
        key_data=json.dumps(credentials_dict)
    )
    ee.Initialize(credentials)
    EE_INITIALIZED = True
    print("✅ Earth Engine initialized successfully")
except Exception as e:
    print(f"⚠️ Earth Engine initialization failed: {str(e)}")
    EE_INITIALIZED = False

# Initialize OpenAI client
openai_client = OpenAI(
    base_url="https://agent-bf02b5dccc2e9e7ca560-lqs5m.ondigitalocean.app/api/v1/",
    api_key="9bosBybqs5DUnKV9Z6FNBxmQG4b7VIwG",
)

# Add new function to generate downloadable report
def generate_downloadable_report(data: Dict[str, Any], location: List[float], messages: List[Dict[str, Any]]) -> str:
    """Generate a detailed report in markdown format for download."""
    try:
        report = f"""# Agricultural Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Location Details
- Latitude: {location[1]}°N
- Longitude: {location[0]}°E

## Vegetation Analysis
### NDVI (Normalized Difference Vegetation Index)
- Value: {data.get('ndvi', 'N/A')}
- Status: {get_ndvi_status(data.get('ndvi', 0))}
- Date: {data.get('dates', {}).get('sentinel2', 'N/A')}

### NDWI (Normalized Difference Water Index)
- Value: {data.get('ndwi', 'N/A')}
- Status: {get_ndwi_status(data.get('ndwi', 0))}

### Soil Analysis
- SAVI: {data.get('savi', 'N/A')}
- Soil Moisture: {data.get('soil_moisture', 'N/A')}

## Weather Forecast
"""
        if data.get('weather'):
            weather = data['weather']
            report += f"""- Temperature Range: {weather['daily']['temperature_2m_min'][0]}°C to {weather['daily']['temperature_2m_max'][0]}°C
- Precipitation: {weather['daily']['precipitation_sum'][0]}mm
- Wind Speed: {weather['daily']['windspeed_10m_max'][0]}m/s

"""

        report += """## Analysis History\n\n"""
        for msg in messages:
            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                report += f"{msg.get('content', '')}\n\n"
            elif isinstance(msg, str):
                report += f"{msg}\n\n"

        return report
    except Exception as e:
        print(f"⚠️ Report generation error: {str(e)}")
        return "Error generating report"

def get_ndvi_status(value: float) -> str:
    if value < 0.2: return "Very Poor"
    if value < 0.4: return "Poor"
    if value < 0.6: return "Moderate"
    if value < 0.8: return "Good"
    return "Excellent"

def get_ndwi_status(value: float) -> str:
    if value < -0.6: return "Very Low"
    if value < -0.2: return "Low"
    if value < 0.2: return "Moderate"
    if value < 0.6: return "High"
    return "Very High"

# Available functions for the AI agent
AVAILABLE_FUNCTIONS = {
    "get_weather": {
        "name": "get_weather",
        "description": "Get weather forecast data for a specific location",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Latitude of the location"},
                "longitude": {"type": "number", "description": "Longitude of the location"}
            },
            "required": ["latitude", "longitude"]
        }
    },
    "get_satellite_data": {
        "name": "get_satellite_data",
        "description": "Get satellite imagery analysis for a specific location",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Latitude of the location"},
                "longitude": {"type": "number", "description": "Longitude of the location"},
                "date": {"type": "string", "description": "Optional date for historical data (YYYY-MM-DD)"}
            },
            "required": ["latitude", "longitude"]
        }
    },
    "geocode": {
        "name": "geocode",
        "description": "Convert a location name to coordinates",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "Name of the location to geocode"}
            },
            "required": ["location"]
        }
    }
}

def geocode(location: str) -> Dict[str, Any]:
    """Convert location name to coordinates"""
    try:
        response = requests.get(f"https://geocode.maps.co/search?q={location}&api_key=67d5a691f0ba6769290722tzy06c90b", timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data:
            raise HTTPException(status_code=404, detail=f"No results found for location: {location}")
        
        return {
            "latitude": float(data[0]['lat']),
            "longitude": float(data[0]['lon']),
            "display_name": data[0]['display_name']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geocoding failed: {str(e)}")

def get_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """Get weather forecast data"""
    try:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ",".join([
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "windspeed_10m_max"
            ]),
            "forecast_days": 7
        }

        response = requests.get('https://api.open-meteo.com/v1/forecast', params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather service error: {str(e)}")

def get_satellite_data(latitude: float, longitude: float, date: Optional[str] = None) -> Dict[str, Any]:
    """Get satellite data using Earth Engine with enhanced analysis"""
    if not EE_INITIALIZED:
        raise HTTPException(status_code=500, detail="Earth Engine not initialized")

    try:
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        # Create a point and buffer it by 1km
        point = ee.Geometry.Point([longitude, latitude])
        buffer = point.buffer(1000)  # 1000 meters = 1km

        startDate = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        endDate = date

        # Get Sentinel-2 data with cloud masking
        def maskS2clouds(image):
            qa = image.select('QA60')
            cloudBitMask = 1 << 10
            cirrusBitMask = 1 << 11
            mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
                   qa.bitwiseAnd(cirrusBitMask).eq(0))
            return image.updateMask(mask)

        s2Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(buffer) \
            .filterDate(startDate, endDate) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .map(maskS2clouds) \
            .sort('system:time_start', False)

        s2Count = s2Collection.size().getInfo()
        if s2Count == 0:
            return {
                "ndvi": 0,
                "ndwi": 0,
                "soil_moisture": 0,
                "dates": {
                    "sentinel2": date,
                    "sentinel1": date
                },
                "map_url": None
            }

        # Get the most recent clear image
        s2Img = s2Collection.first()

        # Calculate vegetation indices
        ndvi = s2Img.normalizedDifference(['B8', 'B4'])  # NIR and Red
        ndwi = s2Img.normalizedDifference(['B8', 'B11'])  # NIR and SWIR
        
        # Additional indices
        savi = s2Img.expression(
            '(NIR - RED) * (1 + L) / (NIR + RED + L)',
            {
                'NIR': s2Img.select('B8'),
                'RED': s2Img.select('B4'),
                'L': 0.5
            }
        )

        # Get Sentinel-1 data
        s1Collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(buffer) \
            .filterDate(startDate, endDate) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .sort('system:time_start', False)

        s1Count = s1Collection.size().getInfo()
        
        # Calculate values for 1km buffer
        ndviValue = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffer,
            scale=10,
            maxPixels=1e9
        ).getInfo().get('nd', 0)  # Default to 0 if no data

        ndwiValue = ndwi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffer,
            scale=10,
            maxPixels=1e9
        ).getInfo().get('nd', 0)  # Default to 0 if no data

        saviValue = savi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffer,
            scale=10,
            maxPixels=1e9
        ).getInfo().get('nd', 0)  # Default to 0 if no data

        soilMoistureValue = 0
        if s1Count > 0:
            s1Img = s1Collection.first()
            soilMoistureValue = s1Img.select('VV').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer,
                scale=10,
                maxPixels=1e9
            ).getInfo().get('VV', 0)
            soilMoistureValue = (soilMoistureValue + 30) / 35  # Normalize soil moisture value

        # Create enhanced visualization with error handling
        try:
            Map = geemap.Map(center=[latitude, longitude], zoom=14)
            Map.addLayer(buffer, {'color': '3388ff'}, '1km Buffer')
            
            ndvi_viz = {
                'min': 0,
                'max': 1,
                'palette': ['red', 'yellow', 'green']
            }
            ndvi_clipped = ndvi.clip(buffer)
            Map.addLayer(ndvi_clipped, ndvi_viz, 'NDVI')

            ndwi_viz = {
                'min': -1,
                'max': 1,
                'palette': ['red', 'yellow', 'blue']
            }
            ndwi_clipped = ndwi.clip(buffer)
            Map.addLayer(ndwi_clipped, ndwi_viz, 'NDWI')

            false_color = s2Img.select(['B8', 'B4', 'B3']).clip(buffer)
            Map.addLayer(false_color, {'min': 0, 'max': 3000}, 'False Color')

            Map.addLayerControl()
            Map.add_legend(title='Vegetation Health', legend_dict={
                'Very Poor (0.0-0.2)': '#FF0000',
                'Poor (0.2-0.4)': '#FF9900',
                'Moderate (0.4-0.6)': '#FFFF00',
                'Good (0.6-0.8)': '#99FF00',
                'Excellent (0.8-1.0)': '#00FF00'
            })

            map_filename = f"static/crop_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            Map.save(os.path.join(STATIC_DIR, os.path.basename(map_filename)))
            map_url = f"/static/{os.path.basename(map_filename)}"
        except Exception as map_error:
            print(f"⚠️ Map generation error: {str(map_error)}")
            map_url = None

        return {
            "ndvi": ndviValue,
            "ndwi": ndwiValue,
            "savi": saviValue,
            "soil_moisture": soilMoistureValue,
            "dates": {
                "sentinel2": s2Img.date().format('YYYY-MM-dd').getInfo(),
                "sentinel1": s1Collection.first().date().format('YYYY-MM-dd').getInfo() if s1Count > 0 else None
            },
            "map_url": map_url
        }
    except Exception as e:
        print(f"⚠️ Satellite data error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_advice(message: str, context: List[Union[str, Dict[str, Any]]], satellite_data: Dict[str, Any], weather_data: Dict[str, Any]) -> str:
    """Generate AI-powered agricultural advice based on satellite and weather data."""
    try:
        ndvi = satellite_data.get('ndvi', 'No data')
        ndwi = satellite_data.get('ndwi', 'No data')
        savi = satellite_data.get('savi', 'No data')
        soil_moisture = satellite_data.get('soil_moisture', 'No data')
        dates = satellite_data.get('dates', {})

        daily = weather_data.get('daily', {})
        precipitation = daily.get('precipitation_sum', [])
        temp_max = daily.get('temperature_2m_max', [])
        temp_min = daily.get('temperature_2m_min', [])
        wind_speed = daily.get('windspeed_10m_max', [])

        ndvi_status = "Very Poor" if ndvi < 0.2 else \
                      "Poor" if ndvi < 0.4 else \
                      "Moderate" if ndvi < 0.6 else \
                      "Good" if ndvi < 0.8 else "Excellent"

        ndwi_status = "Very Low" if ndwi < -0.6 else \
                      "Low" if ndwi < -0.2 else \
                      "Moderate" if ndwi < 0.2 else \
                      "High" if ndwi < 0.6 else "Very High"

        avg_temp = sum([sum(x) for x in zip(temp_max, temp_min)]) / (2 * len(temp_max))
        total_precip = sum(precipitation)

        conversation_history = ""
        for item in context[-5:]:
            if isinstance(item, str):
                conversation_history += f"User: {item}\n"
            elif isinstance(item, dict):
                role = item.get('role', '')
                content = item.get('content', '')
                if role and content:
                    conversation_history += f"{role.capitalize()}: {content}\n"
            else:
                try:
                    conversation_history += f"Context: {str(item)}\n"
                except:
                    pass
        
        prompt = f"""
        You are an expert agricultural advisor specializing in Tunisian agriculture. Analyze the following data and provide detailed, actionable recommendations.

        Previous conversation:
        {conversation_history}

        Current user message: {message}

        Detailed Field Analysis:
        1. Vegetation Health (NDVI):
           - Value: {ndvi}
           - Status: {ndvi_status}
           - Date: {dates.get('sentinel2', 'Not available')}

        2. Water Content (NDWI):
           - Value: {ndwi}
           - Status: {ndwi_status}
           - Implications for irrigation needs

        3. Soil Analysis:
           - SAVI (Soil Adjusted Vegetation Index): {savi}
           - Soil Moisture: {soil_moisture}
           - Implications for root zone conditions

        7-Day Weather Forecast:
        - Daily Precipitation (mm): {precipitation}
        - Temperature Range (°C): {temp_min} to {temp_max}
        - Average Temperature: {avg_temp:.1f}°C
        - Total Expected Precipitation: {total_precip}mm
        - Wind Speed (m/s): {wind_speed}

        Based on this comprehensive analysis:
        1. Evaluate current crop health and stress factors
        2. Provide specific irrigation recommendations
        3. Suggest soil management practices
        4. Recommend timing for agricultural activities
        5. Address any specific concerns from the user's message
        6. Consider local Tunisian agricultural practices and conditions
        7. Provide actionable steps for improvement

        Format your response with clear sections, using markdown for better readability.
        Include both immediate actions and long-term recommendations.
        """

        response = openai_client.chat.completions.create(
            model="n/a",
            messages=[{
                "role": "system",
                "content": "You are an expert agricultural advisor with deep knowledge of Tunisian farming practices, crop science, and precision agriculture."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=12000
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ AI recommendations error: {str(e)}")
        return "I apologize, but I'm having trouble generating recommendations right now. Please try asking your question again."

def decide_action(message: str, context: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Determine which action to take based on user input."""
    try:
        if not isinstance(message, str):
            message = str(message)
            
        report_keywords = ['report', 'download', 'save', 'export', 'generate report']
        is_report_request = any(keyword in message.lower() for keyword in report_keywords)
        
        if is_report_request:
            return {
                "action": "generate_report",
                "parameters": {}
            }
        
        location_keywords = ['analyze location', 'new location', 'different area', 'check another', 'analyze another']
        is_location_request = any(keyword in message.lower() for keyword in location_keywords)
        
        weather_keywords = ['weather forecast', 'weather update', 'check weather', 'updated weather']
        is_weather_request = any(keyword in message.lower() for keyword in weather_keywords)
        
        if not (is_location_request or is_weather_request):
            return {
                "action": "continue_conversation",
                "parameters": {"message": message}
            }
            
        functions = list(AVAILABLE_FUNCTIONS.values())
        
        sanitized_context = []
        for item in context:
            if isinstance(item, str):
                sanitized_context.append(item)
            elif isinstance(item, dict) and 'content' in item:
                sanitized_context.append(f"{item.get('role', 'user')}: {item['content']}")
            else:
                sanitized_context.append(str(item))
        
        prompt = f"""
        You are an AI agent specializing in agricultural analysis. Based on the user's message and context,
        determine which function would be most appropriate to call.

        Available functions:
        {json.dumps(functions, indent=2)}

        User message: {message}
        Previous context: {sanitized_context[-3:]}  # Only use last 3 items

        Determine the most appropriate function to call and its parameters.
        """

        response = openai_client.chat.completions.create(
            model="n/a",
            messages=[{
                "role": "system",
                "content": "You are an AI agent that helps analyze agricultural data and make decisions."
            }, {
                "role": "user",
                "content": prompt
            }],
            functions=functions,
            function_call="auto",
            temperature=0.7
        )

        if response.choices[0].message.function_call:
            function_call = response.choices[0].message.function_call
            return {
                "action": function_call.name,
                "parameters": json.loads(function_call.arguments)
            }
        else:
            return {
                "action": "continue_conversation",
                "parameters": {"message": message}
            }
    except Exception as e:
        print(f"⚠️ Action decision error: {str(e)}")
        return {
            "action": "continue_conversation",
            "parameters": {"message": message}
        }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_location(request: LocationRequest):
    try:
        print(f"Received request: isInitialRequest={request.isInitialRequest}, message={request.message}")
        
        # Handle case when user selects a new location on the map
        if request.coordinates and not request.isInitialRequest and "new location selected" in request.message.lower():
            print(f"Processing new location selection: {request.coordinates}")
            longitude, latitude = request.coordinates
            
            weather_data = get_weather(latitude, longitude)
            satellite_data = get_satellite_data(latitude, longitude)
            
            advice = generate_advice(
                "Please provide a comprehensive analysis of this newly selected agricultural area.",
                request.context or [], 
                satellite_data, 
                weather_data
            )

            return {
                "message": f"I've analyzed the new area at {latitude:.4f}°N, {longitude:.4f}°E.",
                "ndvi": satellite_data["ndvi"],
                "ndwi": satellite_data["ndwi"],
                "soil_moisture": satellite_data["soil_moisture"],
                "weather": weather_data,
                "map_url": satellite_data.get("map_url"),
                "advice": advice
            }
        
        if request.isInitialRequest:
            if request.coordinates:
                # Frontend sends coordinates as [longitude, latitude]
                longitude, latitude = request.coordinates
            elif request.location:
                geocode_result = geocode(request.location)
                latitude = geocode_result["latitude"]
                longitude = geocode_result["longitude"]
            else:
                raise HTTPException(status_code=400, detail="Either location or coordinates must be provided")

            weather_data = get_weather(latitude, longitude)
            satellite_data = get_satellite_data(latitude, longitude)
            
            advice = generate_advice(
                "Please provide a comprehensive analysis of this agricultural area with detailed recommendations.",
                request.context or [], 
                satellite_data, 
                weather_data
            )

            return {
                "message": f"I've analyzed the area at {latitude:.4f}°N, {longitude:.4f}°E.",
                "ndvi": satellite_data["ndvi"],
                "ndwi": satellite_data["ndwi"],
                "soil_moisture": satellite_data["soil_moisture"],
                "weather": weather_data,
                "map_url": satellite_data.get("map_url"),
                "advice": advice
            }
        else:
            if not request.existingData:
                raise HTTPException(status_code=400, detail="No existing analysis data found")

            processed_context = []
            for item in request.context:
                if isinstance(item, str):
                    processed_context.append(item)
                elif isinstance(item, dict):
                    if 'content' in item:
                        if 'role' in item:
                            processed_context.append({"role": item["role"], "content": item["content"]})
                        else:
                            processed_context.append({"role": "user", "content": item["content"]})
                    else:
                        processed_context.append(str(item))
                else:
                    processed_context.append(str(item))
            
            decision = decide_action(request.message, processed_context)
            action = decision["action"]
            parameters = decision["parameters"]
            
            print(f"Decision: action={action}")

            if action == "generate_report":
                if not (request.existingData and request.coordinates):
                    raise HTTPException(status_code=400, detail="Missing data for report generation")
                
                report = generate_downloadable_report(
                    request.existingData,
                    request.coordinates,
                    [{"role": m["role"], "content": m["content"]} for m in request.context or []]
                )
                
                filename = f"agricultural_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                filepath = os.path.join(STATIC_DIR, filename)
                
                with open(filepath, 'w') as f:
                    f.write(report)
                
                return {
                    "message": "Report generated successfully",
                    "ndvi": request.existingData.get("ndvi"),
                    "ndwi": request.existingData.get("ndwi"),
                    "soil_moisture": request.existingData.get("soil_moisture"),
                    "weather": request.existingData.get("weather"),
                    "map_url": request.existingData.get("map_url"),
                    "advice": f"""# Report Generated Successfully 📄

I've created a detailed report of the analysis. You can download it here:

[Download Report](/static/{filename})

The report includes:
- Complete location analysis
- Vegetation health metrics (NDVI, NDWI)
- Soil moisture data
- Weather forecast
- Analysis history

Would you like me to explain any specific part of the report in more detail?"""
                }
            
            elif action == "geocode":
                location = parameters.get("location", request.location)
                if not location:
                    raise HTTPException(status_code=400, detail="Location is required for geocoding")
                result = geocode(location)
                latitude, longitude = result["latitude"], result["longitude"]
                
                weather_data = get_weather(latitude, longitude)
                satellite_data = get_satellite_data(latitude, longitude)
                advice = generate_advice(request.message, request.context or [], satellite_data, weather_data)
                
                return {
                    "message": f"Analyzed new location: {result['display_name']}",
                    "ndvi": satellite_data["ndvi"],
                    "ndwi": satellite_data["ndwi"],
                    "soil_moisture": satellite_data["soil_moisture"],
                    "weather": weather_data,
                    "map_url": satellite_data.get("map_url"),
                    "advice": advice
                }
            
            elif action == "get_weather":
                if request.coordinates:
                    longitude, latitude = request.coordinates
                else:
                    latitude = parameters.get("latitude")
                    longitude = parameters.get("longitude")
                
                if not (latitude and longitude):
                    raise HTTPException(status_code=400, detail="Coordinates are required for weather data")
                
                weather_data = get_weather(latitude, longitude)
                advice = generate_advice(
                    request.message,
                    request.context or [],
                    request.existingData,
                    weather_data
                )
                
                return {
                    "message": "Updated weather analysis",
                    "ndvi": request.existingData.get("ndvi"),
                    "ndwi": request.existingData.get("ndwi"),
                    "soil_moisture": request.existingData.get("soil_moisture"),
                    "weather": weather_data,
                    "map_url": request.existingData.get("map_url"),
                    "advice": advice
                }
            
            else:
                advice = generate_advice(
                    request.message,
                    request.context or [],
                    request.existingData,
                    request.existingData.get("weather", {})
                )
                
                return {
                    "message": "",
                    "ndvi": request.existingData.get("ndvi"),
                    "ndwi": request.existingData.get("ndwi"),
                    "soil_moisture": request.existingData.get("soil_moisture"),
                    "weather": request.existingData.get("weather"),
                    "map_url": request.existingData.get("map_url"),
                    "advice": advice
                }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"⚠️ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
