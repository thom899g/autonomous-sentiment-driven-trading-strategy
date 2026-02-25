"""
Firebase Client for Real-time State Management
Architectural Choice: Firestore provides real-time synchronization across
distributed components and persists state during system restarts.
"""
import os
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import Client as FirestoreClient
from google.cloud.firestore_v1.document import DocumentReference

logger = logging.getLogger(__name__)

@dataclass
class SentimentRecord:
    """Data model for sentiment records"""
    symbol: str
    source: str
    sentiment_score: float
    confidence: float
    raw_text: str
    timestamp: datetime
    metadata: Dict[str, Any]

class FirebaseManager:
    """Manages Firebase connections and operations with error handling"""
    
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._initialize_firebase()
    
    def _initialize_firebase(self) -> None:
        """Initialize Firebase with proper error handling"""
        try:
            # Check for service account file
            service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT")
            if not service_account_path:
                logger.error("FIREBASE_SERVICE_ACCOUNT environment variable not set")
                raise ValueError("Missing Firebase service account configuration")
            
            if not os.path.exists(service_account_path):
                logger.error(f"Service account file not found: {service_account_path}")
                raise FileNotFoundError(f"Service account file not found: {service_account_path}")
            
            # Initialize Firebase app if not already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred)
                logger.info("Firebase initialized successfully")
            
            self._client = firestore.client()
            self._test_connection()
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}", exc_info=True)
            raise
    
    def _test_connection(self) -> bool:
        """Test Firebase connection"""
        try:
            test_ref = self._client.collection("connection_test").document("test")
            test_ref.set({"timestamp": datetime.now()})
            test_ref.delete()
            logger.debug("Firebase connection test successful")
            return True
        except Exception as e:
            logger.error(f"Firebase connection test failed: {e}")
            return False
    
    @property
    def client(self) -> FirestoreClient:
        """Get Firestore client with validation"""
        if self._client is None:
            raise ConnectionError("Firebase client not initialized")
        return self._client
    
    def store_sentiment(self, record: SentimentRecord) -> str:
        """
        Store sentiment record in Firestore with error handling
        
        Args:
            record: SentimentRecord to store
            
        Returns:
            Document ID if successful, empty string on failure
        """
        try:
            collection = self.client.collection("market_sentiment")
            
            # Add validation
            if not record.symbol or record.sentiment