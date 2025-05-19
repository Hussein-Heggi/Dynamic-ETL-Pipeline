import os
import faiss
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
from openai import OpenAI
import torch
import json
import re

class endpointMatcher:
    """
    A class that matches user prompts to API endpoints using:
    - GPT for keyword extraction
    - FAISS + FinBERT for semantic similarity
    - Supports API-endpoint mapping structure
    """
    
    def __init__(self, api_endpoint_map, embedding_model="ProsusAI/finbert"):
        """
        Initialize with the API-endpoint mapping structure
        
        Args:
            api_endpoint_map: Dictionary mapping APIs to their endpoints and descriptions
            embedding_model: Transformer model to use
        """
        self.api_endpoint_map = api_endpoint_map
        
        # Build unified endpoint index across all APIs
        self.endpoint_info = self.build_endpoint_index()
        self.endpoint_names = list(self.endpoint_info.keys())
        self.endpoint_texts = [info['description'] for info in self.endpoint_info.values()]
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        
        # Build FAISS index
        self.build_faiss_index()

    def build_endpoint_index(self):
        """
        Create a unified index of all endpoints with their metadata
        Returns dictionary with structure:
        {
            "endpoint_name": {
                "description": "...",
                "apis": ["api1", "api2", ...]
            }
        }
        """
        endpoint_index = {}
        
        for api_name, api_data in self.api_endpoint_map.items():
            for endpoint in api_data['endpoints']:
                if endpoint not in endpoint_index:
                    endpoint_index[endpoint] = {
                        'description': api_data['endpoint_descriptions'][endpoint],
                        'apis': []
                    }
                endpoint_index[endpoint]['apis'].append(api_name)
                
        return endpoint_index

    def generate_embeddings(self, texts):
        """Generate embeddings using transformers model"""
        # Tokenize input
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Perform mean pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # Create attention mask with the same dimensions as token_embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings along axis 1, but ignore padding tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Sum the mask along axis 1
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Calculate mean
        mean_embeddings = sum_embeddings / sum_mask
        
        # Convert to numpy and normalize
        embeddings = mean_embeddings.numpy()
        embeddings = normalize(embeddings)
        
        return embeddings

    def build_faiss_index(self, index_type="flat", nlist=10):
        """Build FAISS index for endpoint embeddings"""
        self.endpoint_embeddings = self.generate_embeddings(self.endpoint_texts)
        d = self.endpoint_embeddings.shape[1]

        if index_type == "flat":
            self.index = faiss.IndexFlatIP(d)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.endpoint_embeddings)
            self.index.nprobe = min(nlist, 10)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
            
        self.index.add(self.endpoint_embeddings)

    def extract_keywords_with_gpt(self, prompt):
        """Extract relevant keywords from prompt using GPT"""
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable not set.")

        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

        system_prompt = (
            "you are assisting me with extracting keywords from prompts related to finance."
            "Extract relevant data fields or keywords from user requests."
            "Return only a Python list of keywords, e.g., ['keyword1', 'keyword2']."
            "Extract as many key words as possible."
            "if the timeline of such data is implied withing the prompt is implied then output keywords related to it."
        )

        response = client.chat.completions.create(
            model="sonar",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=128,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        # Clean the response more aggressively
        if '[' in content and ']' in content:
            content = content[content.index('['):content.rindex(']')+1]
        try:
            keywords = eval(content)
            if not isinstance(keywords, list):
                keywords = [content]
        except:
            keywords = [word.strip().strip("'\"") 
                    for word in content.strip("[]").split(',')]
        
        return [kw for kw in keywords if kw]  # Remove empty strings

    def semantic_match_faiss(self, phrases):
        """Find matching endpoints using semantic similarity
        Returns dictionary where each endpoint appears only once with its highest score
        """
        phrase_embeddings = self.generate_embeddings(phrases)
        D, I = self.index.search(phrase_embeddings, k=1)
        
        # First pass - collect all matches
        all_matches = {}
        for i, phrase in enumerate(phrases):
            score = D[i][0]
            if score > 0.5:  # Similarity threshold
                matched_endpoint = self.endpoint_names[I[i][0]]
                if matched_endpoint not in all_matches or score > all_matches[matched_endpoint]['score']:
                    all_matches[matched_endpoint] = {
                        'phrase': phrase,
                        'score': round(score, 2),
                        'apis': self.endpoint_info[matched_endpoint]['apis']
                    }
        
        # Convert to phrase-keyed dictionary (optional)
        matches = {
            match['phrase']: {
                'endpoint': endpoint,
                'score': match['score'],
                'apis': match['apis']
            }
            for endpoint, match in all_matches.items()
        }
        
        return matches

    def match_prompt(self, prompt, target_api=None):
        print(f"\n Prompt: {prompt}")
        
        # Extract keywords
        keywords = self.extract_keywords_with_gpt(prompt)
        print(f" GPT Keywords: {keywords}")
        
        # Get semantic matches
        sem_matches = self.semantic_match_faiss(keywords)
        
        # Filter by target API if specified
        if target_api:
            sem_matches = {
                k: v for k, v in sem_matches.items() 
                if target_api in v['apis']
            }
        
        print(f"\n endpoint Matches: {sem_matches}")
        
        # Create list of recommended APIs
        recommended_apis = set()
        api_endpoint_map = {}
        
        for match in sem_matches.values():
            for api in match['apis']:
                recommended_apis.add(api)
                if api not in api_endpoint_map:
                    api_endpoint_map[api] = []
                api_endpoint_map[api].append({
                    'endpoint': match['endpoint'],
                    'score': match['score']
                })
        
        # Extract parameters for matched endpoints
        endpoint_params = self.endpoint_parameters(keywords, api_endpoint_map)
        print(f"\n Extracted Parameters: {endpoint_params}")
        
        return sem_matches, list(recommended_apis), api_endpoint_map, endpoint_params

    def endpoint_parameters(self, keywords, api_endpoint_map):
        """
        Complete working version with all required methods
        """
        temporal_info = self.extract_temporal_info(keywords)
        
        # Prepare base context without temporal phrases
        base_context = [kw for kw in keywords 
                    if not any(d in kw for d in temporal_info['exact_dates'])]
        
        if temporal_info['description']:
            base_context = [kw for kw in base_context 
                        if temporal_info['description'].lower() not in kw.lower()]
        
        endpoint_params = {}
        
        for api_name, endpoints in api_endpoint_map.items():
            for endpoint_info in endpoints:
                endpoint_name = endpoint_info['endpoint']
                
                try:
                    param_schema = self.api_endpoint_map[api_name]['endpoint_params'][endpoint_name]
                except KeyError:
                    continue
                    
                # Handle all endpoints with standard parameter extraction
                params = self._extract_endpoint_params(
                    endpoint_name,
                    param_schema,
                    base_context,
                    temporal_info,
                    api_name
                )
                if params:
                    endpoint_params.setdefault(endpoint_name, {})[api_name] = params
        
        return endpoint_params

    def _extract_endpoint_params(self, endpoint_name, param_schema, context, temporal_info, api_name):
        """More robust parameter extraction with validation"""
        params = {}
        
        # Get LLM response
        response = self._get_llm_response(endpoint_name, param_schema, " ".join(context), api_name)
        
        if response:
            parsed = self._parse_params_from_response(response, param_schema)
            if parsed:
                # Validate required parameters exist
                for req_param in param_schema['required']:
                    if req_param not in parsed:
                        print(f"Warning: Missing required parameter {req_param} for {endpoint_name}")
                        return None
                        
                # Only include parameters that are in the schema
                valid_params = {}
                for param in param_schema['required'] + param_schema['optional']:
                    if param in parsed:
                        valid_params[param] = parsed[param]
                
                params.update(valid_params)
        
        # Add temporal parameters if needed
        if endpoint_name.startswith('TIME_SERIES'):
            if temporal_info['is_relative'] and 'outputsize' in param_schema['optional']:
                params['outputsize'] = 'compact'
        
        return params if params else None

    def _get_llm_response(self, endpoint_name, param_schema, context, api_name):
        """
        Get parameter extraction response from LLM
        """
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            return None
            
        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        
        system_prompt = (
            f"Extract ONLY JSON parameters for {api_name} {endpoint_name} from: {context}\n"
            f"Required parameters: {param_schema['required']}\n"
            f"Optional parameters: {param_schema['optional']}\n"
            "Return the parameters format according to that stated in each api's documentation.\n"
            "DO NOT include any explanations or text outside the JSON object."
        )
        
        try:
            response = client.chat.completions.create(
                model="sonar",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=256,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error for {endpoint_name}: {str(e)}")
            return None

    def extract_temporal_info(self, keywords):
        """
        Extract temporal information from keywords
        """
        temporal_info = {
            'description': '',
            'is_relative': False,
            'exact_dates': [],
            'days': None,
            'temporal_phrases': []
        }
        
        date_pattern = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{4}"
        
        for kw in keywords:
            kw_lower = kw.lower()
            # Check for exact dates
            if re.search(date_pattern, kw, re.IGNORECASE):
                temporal_info['exact_dates'].append(kw)
                temporal_info['temporal_phrases'].append(kw)
            
            # Check for relative time
            elif "last" in kw_lower or "past" in kw_lower:
                match = re.search(r"last\s+(\d+)\s+days", kw_lower)
                if match:
                    temporal_info.update({
                        'description': kw,
                        'is_relative': True,
                        'temporal_phrases': [kw],
                        'days': int(match.group(1))
                    })
        
        return temporal_info

    def _parse_params_from_response(self, response, param_schema):
        """More robust JSON parsing with multiple fallback strategies"""
        if not response:
            return None
            
        try:
            # First try direct JSON parse
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                # Try extracting JSON from markdown code blocks
                if '```json' in response:
                    json_str = response.split('```json')[1].split('```')[0]
                    return json.loads(json_str)
                elif '```' in response:
                    json_str = response.split('```')[1].split('```')[0]
                    return json.loads(json_str)
                    
                # Try extracting just the first {...} block
                if '{' in response and '}' in response:
                    json_str = response[response.index('{'):response.rindex('}')+1]
                    return json.loads(json_str)
                    
                # Final fallback - try to construct basic JSON from key-value pairs
                params = {}
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                for line in lines:
                    if ':' in line:
                        key, val = line.split(':', 1)
                        params[key.strip()] = val.strip().strip('"\'')
                return params if params else None
                
            except Exception as e:
                print(f"Failed to parse response after multiple attempts: {str(e)}")
                return None

if __name__ == "__main__":
    api_endpoint_map = {
        "alpha_vantage_api": {
            "endpoints": [
                "TIME_SERIES_INTRADAY",
                "TIME_SERIES_DAILY",
                "TIME_SERIES_DAILY_ADJUSTED",
                "TIME_SERIES_WEEKLY",
                "TIME_SERIES_WEEKLY_ADJUSTED",
                "TIME_SERIES_MONTHLY",
                "TIME_SERIES_MONTHLY_ADJUSTED",
                "GLOBAL_QUOTE",
                "SYMBOL_SEARCH",
                "CURRENCY_EXCHANGE_RATE",
                "FX_INTRADAY",
                "FX_DAILY",
                "FX_WEEKLY",
                "FX_MONTHLY",
                "SMA",
                "EMA",
                "MACD",
                "RSI",
                "BBANDS",
                "VWAP"
            ],
            "endpoint_descriptions": {
                "TIME_SERIES_INTRADAY": "Get intraday time series (1min, 5min, 15min, 30min, 60min intervals)",
                "TIME_SERIES_DAILY": "Daily historical prices (last 20+ years)",
                "TIME_SERIES_DAILY_ADJUSTED": "Daily prices adjusted for splits/dividends",
                "TIME_SERIES_WEEKLY": "Weekly historical prices",
                "TIME_SERIES_WEEKLY_ADJUSTED": "Weekly prices adjusted for splits/dividends",
                "TIME_SERIES_MONTHLY": "Monthly historical prices",
                "TIME_SERIES_MONTHLY_ADJUSTED": "Monthly prices adjusted for splits/dividends",
                "GLOBAL_QUOTE": "Latest price and volume data",
                "SYMBOL_SEARCH": "Search for ticker symbols and company names",
                "CURRENCY_EXCHANGE_RATE": "Real-time forex rates",
                "FX_INTRADAY": "Intraday foreign exchange rates",
                "FX_DAILY": "Daily foreign exchange rates",
                "FX_WEEKLY": "Weekly foreign exchange rates",
                "FX_MONTHLY": "Monthly foreign exchange rates",
                "SMA": "Simple Moving Average technical indicator",
                "EMA": "Exponential Moving Average technical indicator",
                "MACD": "Moving Average Convergence Divergence technical indicator",
                "RSI": "Relative Strength Index technical indicator",
                "BBANDS": "Bollinger Bands technical indicator",
                "VWAP": "Volume Weighted Average Price (intraday only)"
            },
            "endpoint_params": {
                "TIME_SERIES_INTRADAY": {
                    "required": ["symbol", "interval"],
                    "optional": ["outputsize", "datatype", "adjusted", "extended_hours", "month", "outputsize"]
                },
                "TIME_SERIES_DAILY": {
                    "required": ["symbol"],
                    "optional": ["outputsize", "datatype"]
                },
                "TIME_SERIES_DAILY_ADJUSTED": {
                    "required": ["symbol"],
                    "optional": ["outputsize", "datatype"]
                },
                "TIME_SERIES_WEEKLY": {
                    "required": ["symbol"],
                    "optional": ["datatype"]
                },
                "TIME_SERIES_WEEKLY_ADJUSTED": {
                    "required": ["symbol"],
                    "optional": ["datatype"]
                },
                "TIME_SERIES_MONTHLY": {
                    "required": ["symbol"],
                    "optional": ["datatype"]
                },
                "TIME_SERIES_MONTHLY_ADJUSTED": {
                    "required": ["symbol"],
                    "optional": ["datatype"]
                },
                "GLOBAL_QUOTE": {
                    "required": ["symbol"],
                    "optional": ["datatype"]
                },
                "SYMBOL_SEARCH": {
                    "required": ["keywords"],
                    "optional": ["datatype"]
                },
                "CURRENCY_EXCHANGE_RATE": {
                    "required": ["from_currency", "to_currency"],
                    "optional": ["datatype"]
                },
                "FX_INTRADAY": {
                    "required": ["from_symbol", "to_symbol", "interval"],
                    "optional": ["outputsize", "datatype"]
                },
                "FX_DAILY": {
                    "required": ["from_symbol", "to_symbol"],
                    "optional": ["outputsize", "datatype"]
                },
                "FX_WEEKLY": {
                    "required": ["from_symbol", "to_symbol"],
                    "optional": ["datatype"]
                },
                "FX_MONTHLY": {
                    "required": ["from_symbol", "to_symbol"],
                    "optional": ["datatype"]
                },
                "SMA": {
                    "required": ["symbol", "interval", "time_period", "series_type"],
                    "optional": ["datatype"]
                },
                "EMA": {
                    "required": ["symbol", "interval", "time_period", "series_type"],
                    "optional": ["datatype"]
                },
                "MACD": {
                    "required": ["symbol", "interval", "series_type"],
                    "optional": ["fastperiod", "slowperiod", "signalperiod", "datatype"]
                },
                "RSI": {
                    "required": ["symbol", "interval", "time_period", "series_type"],
                    "optional": ["datatype"]
                },
                "BBANDS": {
                    "required": ["symbol", "interval", "time_period", "series_type"],
                    "optional": ["nbdevup", "nbdevdn", "matype", "datatype"]
                },
                "VWAP": {
                    "required": ["symbol", "interval"],
                    "optional": ["datatype"]
                }
            }
        },
        "polygon_api": {
            "endpoints": [
                "V2_AGGS_TICKER_RANGE",
                "V1_OPEN_CLOSE",
                "V2_AGGS_GROUPED_LOCALE_MARKET_DATE",
                "V2_AGGS_TICKER_PREV",
                "V3_TRADES",
                "V3_QUOTES",
                "V1_LAST_STOCKS",
                "V1_LAST_QUOTE_STOCKS",
                "V2_SNAPSHOT_LOCALE_MARKETS_TICKERS",
                "V2_SNAPSHOT_LOCALE_MARKETS_TICKER",
                "V3_REFERENCE_TICKERS",
                "V3_REFERENCE_TICKER",
                "VX_SIMPLE_MOVINGAVERAGE",
                "VX_TECHNICAL_INDICATORS"
            ],
            "endpoint_descriptions": {
                "V2_AGGS_TICKER_RANGE": "Stock price history with open/high/low/close/volume (custom timeframes: minute, hour, day, week, month)",
                "V1_OPEN_CLOSE": "Daily opening and closing prices for specific date (includes after-hours data)",
                "V2_AGGS_GROUPED_LOCALE_MARKET_DATE": "Market-wide stock prices for specific day (all tickers in exchange)",
                "V2_AGGS_TICKER_PREV": "Previous day's stock market data (closing price and trading volume)",
                "V3_TRADES": "Historical trade-by-trade data (execution prices, sizes, timestamps)",
                "V3_QUOTES": "Historical bid/ask quotes (NBBO market data with timestamps)",
                "V1_LAST_STOCKS": "Latest trade price and details (real-time last transaction)",
                "V1_LAST_QUOTE_STOCKS": "Latest bid/ask prices (real-time market quote)",
                "V2_SNAPSHOT_LOCALE_MARKETS_TICKERS": "Current market prices for all stocks (real-time snapshot)",
                "V2_SNAPSHOT_LOCALE_MARKETS_TICKER": "Current stock market data (price, volume, bid/ask spread)",
                "V3_REFERENCE_TICKERS": "Stock symbol search (ticker lookup and company names)",
                "V3_REFERENCE_TICKER": "Company profile data (financial reference information)",
                "VX_SIMPLE_MOVINGAVERAGE": "Stock technical analysis: Simple Moving Average (SMA values)",
                "VX_TECHNICAL_INDICATORS": "Technical indicators for stocks (RSI, MACD, Bollinger Bands)"
            },
            "endpoint_params": {
                "V2_AGGS_TICKER_RANGE": {
                    "required": ["ticker"],
                    "optional": ["multiplier", "timespan", "from", "to", "adjusted", "sort", "limit"]
                },
                "V1_OPEN_CLOSE": {
                    "required": ["ticker", "date"],
                    "optional": ["adjusted"]
                },
                "V2_AGGS_GROUPED_LOCALE_MARKET_DATE": {
                    "required": ["locale", "market", "date"],
                    "optional": ["adjusted"]
                },
                "V2_AGGS_TICKER_PREV": {
                    "required": ["ticker"],
                    "optional": ["adjusted"]
                },
                "V3_TRADES": {
                    "required": ["ticker"],
                    "optional": ["timestamp", "timestamp.lt", "timestamp.lte", "timestamp.gt", "timestamp.gte", "limit", "order", "sort"]
                },
                "V3_QUOTES": {
                    "required": ["ticker"],
                    "optional": ["timestamp", "timestamp.lt", "timestamp.lte", "timestamp.gt", "timestamp.gte", "limit", "order", "sort"]
                },
                "V1_LAST_STOCKS": {
                    "required": ["ticker"],
                    "optional": []
                },
                "V1_LAST_QUOTE_STOCKS": {
                    "required": ["ticker"],
                    "optional": []
                },
                "V2_SNAPSHOT_LOCALE_MARKETS_TICKERS": {
                    "required": ["locale", "market"],
                    "optional": []
                },
                "V2_SNAPSHOT_LOCALE_MARKETS_TICKER": {
                    "required": ["locale", "market", "ticker"],
                    "optional": []
                },
                "V3_REFERENCE_TICKERS": {
                    "required": [],
                    "optional": ["ticker", "type", "market", "exchange", "cusip", "cik", "date", "search", "active", "sort", "order", "limit"]
                },
                "V3_REFERENCE_TICKER": {
                    "required": ["ticker"],
                    "optional": []
                },
                "VX_SIMPLE_MOVINGAVERAGE": {
                    "required": ["ticker", "window"],
                    "optional": ["timespan", "adjusted", "series_type", "expand_underlying"]
                },
                "VX_TECHNICAL_INDICATORS": {
                    "required": ["ticker", "indicator"],
                    "optional": ["timespan", "window", "series_type", "time_period", "sd", "ma_type"]
                }
            }
        }
    }

    matcher = endpointMatcher(api_endpoint_map, embedding_model="ProsusAI/finbert")

    prompt = "Show me the daily adjusted closing prices, trading volume, and 50-day simple moving average for Apple (APPL) from January 1st to March 31st 2023, along with the most recent quote data."
    endpoint_matches, recommended_apis, api_endpoint_mapping, endpoint_params = matcher.match_prompt(prompt)
    
    # Print results in the requested format
    print("\n=== Recommended APIs ===")
    for api in recommended_apis:
        print(f"- {api}")
    
    print("\n=== Recommended Endpoints for Each API ===")
    for api, endpoints in api_endpoint_mapping.items():
        print(f"\nAPI: {api}")
        for endpoint in endpoints:
            print(f"  - {endpoint['endpoint']} (confidence score: {endpoint['score']})")
    
    print("\n=== Recommended Parameters for Each Endpoint ===")
    if endpoint_params:
        for endpoint, params in endpoint_params.items():
            print(f"\nEndpoint: {endpoint}")
            for api, param_values in params.items():
                print(f"  API: {api}")
                if param_values:  # Only print if parameters exist
                    for param, value in param_values.items():
                        print(f"    - {param}: {value}")
                else:
                    print("    - No parameters extracted")
    else:
        print("No parameters were extracted for any endpoints")
