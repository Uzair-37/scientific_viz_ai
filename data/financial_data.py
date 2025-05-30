"""
Financial data processing and loading utilities.
"""
import os
import logging
import json
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from transformers import AutoTokenizer
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FinancialDataset(Dataset):
    """
    Multi-modal financial dataset for training and evaluation.
    
    Supports loading of time series, text, fundamental, and network data.
    """
    
    def __init__(
        self,
        symbols: Union[List[str], str],
        start_date: str,
        end_date: str,
        time_steps: int = 252,  # One trading year
        modalities: List[str] = ["time_series", "text", "fundamental"],
        features: Optional[List[str]] = None,
        text_max_length: int = 512,
        cache_dir: Optional[str] = None,
        transform=None,
        include_target: bool = True,
        target_horizon: int = 21,  # Trading month
        target_type: str = "return"
    ):
        """
        Initialize the financial dataset.
        
        Args:
            symbols: List of ticker symbols or path to file with symbols
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            time_steps: Number of time steps for time series data
            modalities: List of modalities to include
            features: List of features to include (if None, use all)
            text_max_length: Maximum length for text data
            cache_dir: Directory for caching data
            transform: Transforms to apply to the data
            include_target: Whether to include target variables
            target_horizon: Forecast horizon for target (in trading days)
            target_type: Type of target variable (return, volatility, direction)
        """
        self.modalities = modalities
        self.time_steps = time_steps
        self.text_max_length = text_max_length
        self.transform = transform
        self.include_target = include_target
        self.target_horizon = target_horizon
        self.target_type = target_type
        
        # Parse symbols
        if isinstance(symbols, str) and os.path.exists(symbols):
            with open(symbols, 'r') as f:
                self.symbols = [line.strip() for line in f if line.strip()]
        elif isinstance(symbols, list):
            self.symbols = symbols
        else:
            raise ValueError("symbols must be a list or a path to a file")
        
        # Set up cache
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            
        # Set default features if not provided
        if features is None:
            self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'MovingAvg10', 'RSI', 'MACD']
        else:
            self.features = features
            
        # Initialize tokenizer for text if needed
        self.tokenizer = None
        if "text" in self.modalities:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Load or download data
        self.data = self._load_data(start_date, end_date)
        
        # Create valid indices
        self._create_valid_indices()
        
        logger.info(f"Initialized FinancialDataset with {len(self.valid_indices)} samples")
    
    def _load_data(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Load or download financial data for all symbols.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            Dictionary with loaded data
        """
        data = {
            "time_series": {},
            "text": {},
            "fundamental": {},
            "network": {}
        }
        
        # Check cache first
        if self.cache_dir:
            cache_file = os.path.join(
                self.cache_dir,
                f"financial_data_{start_date}_{end_date}_{len(self.symbols)}.pt"
            )
            
            if os.path.exists(cache_file):
                logger.info(f"Loading data from cache: {cache_file}")
                cached_data = torch.load(cache_file)
                return cached_data
        
        # Download time series data
        if "time_series" in self.modalities:
            logger.info(f"Downloading time series data for {len(self.symbols)} symbols")
            
            # We need to download additional history for features calculation
            extended_start_date = (
                datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=100)
            ).strftime("%Y-%m-%d")
            
            for symbol in self.symbols:
                try:
                    # Download data
                    stock_data = yf.download(
                        symbol,
                        start=extended_start_date,
                        end=end_date,
                        progress=False
                    )
                    
                    if len(stock_data) > 0:
                        # Calculate additional features
                        stock_data = self._calculate_features(stock_data)
                        
                        # Trim to requested date range
                        stock_data = stock_data[stock_data.index >= start_date]
                        
                        # Store data
                        data["time_series"][symbol] = stock_data
                    else:
                        logger.warning(f"No data available for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error downloading data for {symbol}: {e}")
        
        # Get text data (news, filings)
        if "text" in self.modalities:
            logger.info("Loading text data")
            # In a real implementation, this would fetch news and filings
            # For this example, we'll create dummy text data
            for symbol in self.symbols:
                if symbol in data["time_series"]:
                    dates = data["time_series"][symbol].index
                    data["text"][symbol] = self._create_dummy_text_data(symbol, dates)
        
        # Get fundamental data
        if "fundamental" in self.modalities:
            logger.info("Loading fundamental data")
            # In a real implementation, this would fetch fundamental data
            # For this example, we'll create dummy fundamental data
            for symbol in self.symbols:
                if symbol in data["time_series"]:
                    dates = data["time_series"][symbol].index
                    data["fundamental"][symbol] = self._create_dummy_fundamental_data(symbol, dates)
        
        # Create network data
        if "network" in self.modalities:
            logger.info("Creating network data")
            # In a real implementation, this would build relationship networks
            # For this example, we'll create a dummy network
            data["network"] = self._create_dummy_network_data()
        
        # Cache the data
        if self.cache_dir:
            logger.info(f"Caching data to {cache_file}")
            torch.save(data, cache_file)
        
        return data
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional features for time series data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional features
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate moving averages
        df['MovingAvg10'] = df['Close'].rolling(window=10).mean()
        df['MovingAvg30'] = df['Close'].rolling(window=30).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['UpperBand'] = df['MA20'] + (std20 * 2)
        df['LowerBand'] = df['MA20'] - (std20 * 2)
        
        # Calculate volatility
        df['Volatility'] = df['Returns'].rolling(window=21).std() * np.sqrt(252)
        
        # Drop NaN values
        df = df.fillna(method='bfill')
        
        return df
    
    def _create_dummy_text_data(self, symbol: str, dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Create dummy text data for a symbol.
        
        Args:
            symbol: Ticker symbol
            dates: DatetimeIndex of available dates
            
        Returns:
            Dictionary with text data
        """
        # Simulate news and filings
        text_data = {
            "news": {},
            "filings": {}
        }
        
        # Create some dummy news (one per week)
        for i in range(0, len(dates), 5):
            date = dates[i]
            
            # Randomly decide if there's news on this day
            if np.random.random() < 0.3:
                sentiment = np.random.choice(["positive", "neutral", "negative"], p=[0.3, 0.5, 0.2])
                
                if sentiment == "positive":
                    news_text = f"{symbol} reports strong quarterly results above expectations."
                elif sentiment == "neutral":
                    news_text = f"{symbol} announces new product line meeting analyst expectations."
                else:
                    news_text = f"{symbol} warns of potential shortfall in upcoming earnings report."
                
                text_data["news"][date.strftime("%Y-%m-%d")] = {
                    "text": news_text,
                    "sentiment": sentiment
                }
        
        # Create some dummy filings (quarterly)
        for i in range(0, len(dates), 63):  # ~quarterly
            if i < len(dates):
                date = dates[i]
                
                filing_text = f"""
                {symbol} Inc. Financial Results for the Quarter Ending {date.strftime('%B %d, %Y')}
                
                Revenue: ${np.random.randint(100, 1000)} million
                Net Income: ${np.random.randint(10, 100)} million
                EPS: ${np.random.uniform(0.5, 2.5):.2f}
                
                Management Comment: We are {np.random.choice(['pleased with', 'satisfied with', 'concerned about'])} 
                our performance this quarter and expect to {np.random.choice(['grow', 'maintain', 'improve'])} 
                in the coming quarters.
                """
                
                text_data["filings"][date.strftime("%Y-%m-%d")] = {
                    "text": filing_text,
                    "type": "10-Q"
                }
        
        return text_data
    
    def _create_dummy_fundamental_data(self, symbol: str, dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Create dummy fundamental data for a symbol.
        
        Args:
            symbol: Ticker symbol
            dates: DatetimeIndex of available dates
            
        Returns:
            Dictionary with fundamental data
        """
        # Create quarterly fundamental data
        fundamental_data = {}
        
        # Base values that will grow/shrink over time
        base_revenue = np.random.uniform(500, 10000)
        base_assets = np.random.uniform(1000, 20000)
        base_liabilities = base_assets * np.random.uniform(0.3, 0.7)
        base_equity = base_assets - base_liabilities
        growth_rate = np.random.uniform(-0.05, 0.15)  # -5% to 15% growth
        
        # Create quarterly data
        for i in range(0, len(dates), 63):  # ~quarterly
            if i < len(dates):
                date = dates[i]
                quarter_offset = i // 63
                
                # Add some growth and randomness
                revenue = base_revenue * (1 + growth_rate) ** quarter_offset * np.random.uniform(0.9, 1.1)
                profit_margin = np.random.uniform(0.05, 0.25)
                net_income = revenue * profit_margin
                
                assets = base_assets * (1 + growth_rate * 0.8) ** quarter_offset * np.random.uniform(0.95, 1.05)
                liabilities = base_liabilities * (1 + growth_rate * 0.7) ** quarter_offset * np.random.uniform(0.95, 1.05)
                equity = assets - liabilities
                
                # Store fundamental data
                fundamental_data[date.strftime("%Y-%m-%d")] = {
                    "income_statement": {
                        "revenue": revenue,
                        "cost_of_goods_sold": revenue * np.random.uniform(0.4, 0.7),
                        "gross_profit": revenue * np.random.uniform(0.3, 0.6),
                        "operating_expenses": revenue * np.random.uniform(0.2, 0.4),
                        "operating_income": revenue * np.random.uniform(0.1, 0.3),
                        "net_income": net_income
                    },
                    "balance_sheet": {
                        "total_assets": assets,
                        "total_liabilities": liabilities,
                        "total_equity": equity,
                        "cash": assets * np.random.uniform(0.05, 0.2),
                        "debt": liabilities * np.random.uniform(0.3, 0.7)
                    },
                    "ratios": {
                        "pe_ratio": np.random.uniform(10, 30),
                        "price_to_book": np.random.uniform(1, 5),
                        "debt_to_equity": liabilities / max(equity, 1),
                        "roe": net_income / max(equity, 1),
                        "roa": net_income / max(assets, 1),
                        "current_ratio": np.random.uniform(1, 3)
                    }
                }
        
        return fundamental_data
    
    def _create_dummy_network_data(self) -> Dict[str, Any]:
        """
        Create dummy network data for the symbols.
        
        Returns:
            Dictionary with network data
        """
        num_symbols = len(self.symbols)
        
        # Create node features (one row per symbol)
        node_features = np.random.randn(num_symbols, 10)  # 10 features per node
        
        # Create edges (relationships between companies)
        num_edges = int(num_symbols * 3)  # ~3 connections per company on average
        
        edge_index = []
        edge_features = []
        
        for _ in range(num_edges):
            source = np.random.randint(0, num_symbols)
            target = np.random.randint(0, num_symbols)
            
            # Avoid self-loops
            while source == target:
                target = np.random.randint(0, num_symbols)
            
            edge_index.append((source, target))
            
            # Edge features (e.g., relationship type, strength)
            edge_features.append(np.random.randn(5))  # 5 features per edge
        
        # Convert to appropriate format
        network_data = {
            "node_features": np.array(node_features),
            "edge_index": np.array(edge_index).T,  # [2, num_edges]
            "edge_features": np.array(edge_features),
            "node_mapping": {i: symbol for i, symbol in enumerate(self.symbols)}
        }
        
        return network_data
    
    def _create_valid_indices(self):
        """
        Create valid indices for the dataset based on available data.
        """
        self.valid_indices = []
        
        # For each symbol, find valid time points
        for symbol_idx, symbol in enumerate(self.symbols):
            # Skip if time series data not available
            if symbol not in self.data["time_series"]:
                continue
                
            ts_data = self.data["time_series"][symbol]
            
            # We need at least time_steps data points
            if len(ts_data) < self.time_steps:
                continue
            
            # Find valid timepoints (need enough history and future data for target)
            for i in range(self.time_steps, len(ts_data) - self.target_horizon if self.include_target else len(ts_data)):
                date = ts_data.index[i]
                date_str = date.strftime("%Y-%m-%d")
                
                # Check if we have data for all required modalities
                valid = True
                
                if "text" in self.modalities and symbol not in self.data["text"]:
                    valid = False
                
                if "fundamental" in self.modalities and symbol not in self.data["fundamental"]:
                    valid = False
                
                if valid:
                    self.valid_indices.append((symbol_idx, i, date_str))
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index into the dataset
            
        Returns:
            Dictionary with modality data and target
        """
        symbol_idx, time_idx, date_str = self.valid_indices[idx]
        symbol = self.symbols[symbol_idx]
        
        sample = {}
        
        # Get time series data
        if "time_series" in self.modalities:
            ts_data = self.data["time_series"][symbol]
            
            # Extract window of time steps
            window = ts_data.iloc[time_idx - self.time_steps:time_idx]
            
            # Select requested features
            feature_data = window[self.features].values
            
            # Normalize features (simple standardization)
            feature_means = np.nanmean(feature_data, axis=0)
            feature_stds = np.nanstd(feature_data, axis=0)
            feature_stds = np.where(feature_stds == 0, 1, feature_stds)  # Avoid division by zero
            normalized_features = (feature_data - feature_means) / feature_stds
            
            # Replace any remaining NaNs with 0
            normalized_features = np.nan_to_num(normalized_features)
            
            sample["time_series"] = torch.tensor(normalized_features, dtype=torch.float)
        
        # Get text data
        if "text" in self.modalities and symbol in self.data["text"]:
            text_data = self.data["text"][symbol]
            
            # Find most recent news and filings
            date = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Find news within last 30 days
            recent_news = []
            for news_date, news in text_data["news"].items():
                news_datetime = datetime.strptime(news_date, "%Y-%m-%d")
                if 0 <= (date - news_datetime).days <= 30:
                    recent_news.append(news["text"])
            
            # Find most recent filing
            recent_filing = None
            most_recent_date = None
            
            for filing_date, filing in text_data["filings"].items():
                filing_datetime = datetime.strptime(filing_date, "%Y-%m-%d")
                if filing_datetime <= date:
                    if most_recent_date is None or filing_datetime > most_recent_date:
                        most_recent_date = filing_datetime
                        recent_filing = filing["text"]
            
            # Combine text
            combined_text = ""
            
            if recent_news:
                combined_text += "NEWS: " + " ".join(recent_news)
            
            if recent_filing:
                if combined_text:
                    combined_text += " "
                combined_text += "FILING: " + recent_filing
            
            # Tokenize text
            if combined_text and self.tokenizer:
                encoded = self.tokenizer(
                    combined_text,
                    max_length=self.text_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                sample["text"] = {
                    "input_ids": encoded["input_ids"][0],
                    "attention_mask": encoded["attention_mask"][0]
                }
            else:
                # Empty text
                sample["text"] = {
                    "input_ids": torch.zeros(self.text_max_length, dtype=torch.long),
                    "attention_mask": torch.zeros(self.text_max_length, dtype=torch.long)
                }
        
        # Get fundamental data
        if "fundamental" in self.modalities and symbol in self.data["fundamental"]:
            fundamental_data = self.data["fundamental"][symbol]
            
            # Find most recent fundamental data
            date = datetime.strptime(date_str, "%Y-%m-%d")
            most_recent_date = None
            recent_fundamental = None
            
            for fund_date, fund_data in fundamental_data.items():
                fund_datetime = datetime.strptime(fund_date, "%Y-%m-%d")
                if fund_datetime <= date:
                    if most_recent_date is None or fund_datetime > most_recent_date:
                        most_recent_date = fund_datetime
                        recent_fundamental = fund_data
            
            if recent_fundamental:
                # Extract and flatten fundamental features
                income_stmt = [
                    recent_fundamental["income_statement"]["revenue"],
                    recent_fundamental["income_statement"]["gross_profit"],
                    recent_fundamental["income_statement"]["operating_income"],
                    recent_fundamental["income_statement"]["net_income"]
                ]
                
                balance_sheet = [
                    recent_fundamental["balance_sheet"]["total_assets"],
                    recent_fundamental["balance_sheet"]["total_liabilities"],
                    recent_fundamental["balance_sheet"]["total_equity"],
                    recent_fundamental["balance_sheet"]["cash"],
                    recent_fundamental["balance_sheet"]["debt"]
                ]
                
                ratios = [
                    recent_fundamental["ratios"]["pe_ratio"],
                    recent_fundamental["ratios"]["price_to_book"],
                    recent_fundamental["ratios"]["debt_to_equity"],
                    recent_fundamental["ratios"]["roe"],
                    recent_fundamental["ratios"]["roa"],
                    recent_fundamental["ratios"]["current_ratio"]
                ]
                
                # Combine and normalize
                fundamental_features = income_stmt + balance_sheet + ratios
                fundamental_features = np.array(fundamental_features)
                
                # Simple robust normalization
                q1 = np.percentile(fundamental_features, 25)
                q3 = np.percentile(fundamental_features, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Clip outliers and normalize to [0, 1]
                clipped = np.clip(fundamental_features, lower_bound, upper_bound)
                normalized = (clipped - lower_bound) / (upper_bound - lower_bound)
                normalized = np.nan_to_num(normalized, nan=0.5)  # Replace NaNs with 0.5
                
                sample["fundamental"] = {
                    "income_statement": torch.tensor(income_stmt, dtype=torch.float),
                    "balance_sheet": torch.tensor(balance_sheet, dtype=torch.float),
                    "ratios": torch.tensor(ratios, dtype=torch.float),
                    "combined": torch.tensor(normalized, dtype=torch.float)
                }
            else:
                # No fundamental data available
                sample["fundamental"] = {
                    "income_statement": torch.zeros(4, dtype=torch.float),
                    "balance_sheet": torch.zeros(5, dtype=torch.float),
                    "ratios": torch.zeros(6, dtype=torch.float),
                    "combined": torch.zeros(15, dtype=torch.float)
                }
        
        # Get network data
        if "network" in self.modalities:
            network_data = self.data["network"]
            
            # For simplicity, we just include the node features for this symbol
            node_features = torch.tensor(network_data["node_features"][symbol_idx], dtype=torch.float)
            
            # Find connected symbols (neighbors)
            neighbors = []
            edge_indices = np.where(network_data["edge_index"][0] == symbol_idx)[0]
            
            for edge_idx in edge_indices:
                neighbor_idx = network_data["edge_index"][1][edge_idx]
                edge_feat = network_data["edge_features"][edge_idx]
                
                neighbors.append({
                    "symbol": self.symbols[neighbor_idx],
                    "features": torch.tensor(edge_feat, dtype=torch.float)
                })
            
            sample["network"] = {
                "node_features": node_features,
                "neighbors": neighbors
            }
        
        # Add metadata
        sample["metadata"] = {
            "symbol": symbol,
            "date": date_str,
            "symbol_idx": symbol_idx,
            "time_idx": time_idx
        }
        
        # Add target if requested
        if self.include_target:
            ts_data = self.data["time_series"][symbol]
            
            # Extract target value
            if time_idx + self.target_horizon < len(ts_data):
                current_price = ts_data.iloc[time_idx]["Close"]
                future_price = ts_data.iloc[time_idx + self.target_horizon]["Close"]
                
                if self.target_type == "return":
                    target = (future_price / current_price) - 1
                elif self.target_type == "direction":
                    target = 1 if future_price > current_price else 0
                elif self.target_type == "volatility":
                    returns = ts_data.iloc[time_idx:time_idx + self.target_horizon]["Returns"].values
                    target = np.std(returns) * np.sqrt(252 / self.target_horizon)
                else:
                    raise ValueError(f"Unsupported target type: {self.target_type}")
                
                sample["target"] = torch.tensor(target, dtype=torch.float)
            else:
                # Not enough future data, use NaN
                sample["target"] = torch.tensor(float('nan'), dtype=torch.float)
        
        # Apply transforms if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def create_financial_dataloaders(
    symbols: Union[List[str], str],
    start_date: str,
    end_date: str,
    time_steps: int = 252,
    batch_size: int = 32,
    modalities: List[str] = ["time_series", "text", "fundamental"],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    cache_dir: Optional[str] = None,
    num_workers: int = 4,
    target_horizon: int = 21,
    target_type: str = "return",
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders for financial data.
    
    Args:
        symbols: List of ticker symbols or path to file with symbols
        start_date: Start date for data collection (YYYY-MM-DD)
        end_date: End date for data collection (YYYY-MM-DD)
        time_steps: Number of time steps for time series data
        batch_size: Batch size for the data loaders
        modalities: List of modalities to include
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        cache_dir: Directory for caching data
        num_workers: Number of workers for data loading
        target_horizon: Forecast horizon for target (in trading days)
        target_type: Type of target variable (return, volatility, direction)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create the full dataset
    full_dataset = FinancialDataset(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        time_steps=time_steps,
        modalities=modalities,
        cache_dir=cache_dir,
        include_target=True,
        target_horizon=target_horizon,
        target_type=target_type
    )
    
    # Get valid indices
    valid_indices = list(range(len(full_dataset)))
    np.random.shuffle(valid_indices)
    
    # Calculate sizes
    dataset_size = len(valid_indices)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Split indices
    train_indices = valid_indices[:train_size]
    val_indices = valid_indices[train_size:train_size + val_size]
    test_indices = valid_indices[train_size + val_size:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders with {train_size} training, {val_size} validation, and {test_size} test samples")
    
    return train_loader, val_loader, test_loader