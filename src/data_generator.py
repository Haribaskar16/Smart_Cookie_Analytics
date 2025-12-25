"""
Data Generation Module for Smart Cookie Analytics

This module provides utilities for generating synthetic cookie analytics data,
including user interactions, event tracking, and performance metrics.

Author: Smart Cookie Analytics Team
Date: 2025-12-25
Version: 1.0.0
"""

import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Enumeration of possible cookie analytics event types."""
    PAGE_VIEW = "page_view"
    CLICK = "click"
    SCROLL = "scroll"
    FORM_SUBMIT = "form_submit"
    PURCHASE = "purchase"
    ERROR = "error"
    SESSION_START = "session_start"
    SESSION_END = "session_end"


class UserSegment(Enum):
    """Enumeration of user segments for analytics."""
    NEW = "new"
    RETURNING = "returning"
    VIP = "vip"
    INACTIVE = "inactive"


@dataclass
class CookieEvent:
    """
    Data class representing a single cookie analytics event.
    
    Attributes:
        event_id: Unique identifier for the event
        timestamp: When the event occurred
        user_id: Unique identifier for the user
        event_type: Type of event (from EventType enum)
        event_data: Additional event-specific data
        session_id: Session identifier
        user_segment: User segment classification
        page_url: URL where event occurred
        referrer: HTTP referrer
        user_agent: Browser user agent string
    """
    event_id: str
    timestamp: datetime
    user_id: str
    event_type: str
    event_data: Dict[str, Any]
    session_id: str
    user_segment: str
    page_url: str
    referrer: str
    user_agent: str


class DataGenerationError(Exception):
    """Custom exception for data generation errors."""
    pass


class DataGenerator:
    """
    Generate synthetic cookie analytics data for testing and development.
    
    This class provides methods to generate realistic cookie analytics data
    including user events, sessions, and performance metrics.
    """
    
    # Sample data for realistic generation
    PAGE_URLS = [
        "/", "/products", "/products/cookies", "/products/brownies",
        "/about", "/contact", "/cart", "/checkout", "/account",
        "/blog", "/blog/recipes", "/support", "/faq"
    ]
    
    REFERRERS = [
        "google.com", "facebook.com", "twitter.com", "direct",
        "instagram.com", "reddit.com", "pinterest.com", ""
    ]
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)",
        "Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    ]
    
    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the DataGenerator.
        
        Args:
            seed: Optional random seed for reproducible generation
            
        Raises:
            DataGenerationError: If seed is invalid
        """
        try:
            if seed is not None:
                random.seed(seed)
            logger.info(f"DataGenerator initialized with seed: {seed}")
        except Exception as e:
            raise DataGenerationError(f"Failed to initialize DataGenerator: {str(e)}")
    
    def generate_user_id(self) -> str:
        """
        Generate a unique user identifier.
        
        Returns:
            A string user ID in format: user_XXXXXXXXXX
        """
        return f"user_{random.randint(100000000, 999999999)}"
    
    def generate_session_id(self) -> str:
        """
        Generate a unique session identifier.
        
        Returns:
            A string session ID in format: sess_XXXXXXXXXX
        """
        return f"sess_{random.randint(100000000, 999999999)}"
    
    def generate_event_id(self) -> str:
        """
        Generate a unique event identifier.
        
        Returns:
            A string event ID in format: evt_XXXXXXXXXX
        """
        return f"evt_{random.randint(100000000, 999999999)}"
    
    def generate_timestamp(
        self, 
        start_date: Optional[datetime] = None,
        days_back: int = 30
    ) -> datetime:
        """
        Generate a random timestamp within a date range.
        
        Args:
            start_date: Start date for range. Defaults to now
            days_back: Number of days to go back from start_date
            
        Returns:
            A datetime object within the specified range
        """
        if start_date is None:
            start_date = datetime.utcnow()
        
        random_days = random.randint(0, days_back)
        random_seconds = random.randint(0, 86399)
        
        return start_date - timedelta(days=random_days, seconds=random_seconds)
    
    def get_user_segment(self) -> str:
        """
        Randomly assign a user segment based on weighted distribution.
        
        Returns:
            A UserSegment enum value as string
        """
        segments = [s.value for s in UserSegment]
        weights = [0.30, 0.50, 0.15, 0.05]  # NEW, RETURNING, VIP, INACTIVE
        return random.choices(segments, weights=weights, k=1)[0]
    
    def generate_event_data(self, event_type: str) -> Dict[str, Any]:
        """
        Generate event-specific data based on event type.
        
        Args:
            event_type: The type of event from EventType enum
            
        Returns:
            Dictionary containing event-specific data
            
        Raises:
            DataGenerationError: If event_type is invalid
        """
        try:
            if event_type == EventType.PAGE_VIEW.value:
                return {
                    "page_load_time": random.randint(100, 5000),
                    "scroll_depth": random.randint(0, 100)
                }
            elif event_type == EventType.CLICK.value:
                return {
                    "element_id": f"btn_{random.randint(1, 100)}",
                    "element_class": random.choice(["primary", "secondary", "danger"])
                }
            elif event_type == EventType.PURCHASE.value:
                return {
                    "product_id": f"prod_{random.randint(1000, 9999)}",
                    "amount": round(random.uniform(5.99, 199.99), 2),
                    "currency": "USD",
                    "items_count": random.randint(1, 10)
                }
            elif event_type == EventType.FORM_SUBMIT.value:
                return {
                    "form_id": f"form_{random.randint(1, 50)}",
                    "field_count": random.randint(1, 20),
                    "submission_time": random.randint(10, 300)
                }
            elif event_type == EventType.ERROR.value:
                return {
                    "error_code": random.choice([400, 404, 500, 503]),
                    "error_message": random.choice([
                        "Network timeout",
                        "Server error",
                        "Page not found",
                        "Invalid request"
                    ]),
                    "stack_trace": "See error logs for details"
                }
            elif event_type == EventType.SCROLL.value:
                return {
                    "scroll_direction": random.choice(["up", "down"]),
                    "scroll_distance": random.randint(0, 5000)
                }
            else:
                return {}
        except Exception as e:
            raise DataGenerationError(f"Failed to generate event data: {str(e)}")
    
    def generate_single_event(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> CookieEvent:
        """
        Generate a single cookie analytics event.
        
        Args:
            user_id: Optional user ID. If None, a new one is generated
            session_id: Optional session ID. If None, a new one is generated
            timestamp: Optional event timestamp. If None, current time is used
            
        Returns:
            A CookieEvent instance
            
        Raises:
            DataGenerationError: If event generation fails
        """
        try:
            user_id = user_id or self.generate_user_id()
            session_id = session_id or self.generate_session_id()
            timestamp = timestamp or datetime.utcnow()
            
            event_type = random.choice([e.value for e in EventType])
            
            event = CookieEvent(
                event_id=self.generate_event_id(),
                timestamp=timestamp,
                user_id=user_id,
                event_type=event_type,
                event_data=self.generate_event_data(event_type),
                session_id=session_id,
                user_segment=self.get_user_segment(),
                page_url=random.choice(self.PAGE_URLS),
                referrer=random.choice(self.REFERRERS),
                user_agent=random.choice(self.USER_AGENTS)
            )
            
            logger.debug(f"Generated event: {event.event_id}")
            return event
        except Exception as e:
            raise DataGenerationError(f"Failed to generate single event: {str(e)}")
    
    def generate_events(
        self,
        count: int = 100,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[CookieEvent]:
        """
        Generate multiple cookie analytics events.
        
        Args:
            count: Number of events to generate. Must be positive
            user_id: Optional user ID for all events
            session_id: Optional session ID for all events
            
        Returns:
            List of CookieEvent instances
            
        Raises:
            DataGenerationError: If count is invalid or generation fails
        """
        if count <= 0:
            raise DataGenerationError("Event count must be a positive integer")
        
        if count > 10000:
            logger.warning(f"Generating large dataset: {count} events")
        
        try:
            events = []
            for _ in range(count):
                event = self.generate_single_event(
                    user_id=user_id,
                    session_id=session_id
                )
                events.append(event)
            
            logger.info(f"Generated {count} events successfully")
            return events
        except Exception as e:
            raise DataGenerationError(f"Failed to generate {count} events: {str(e)}")
    
    def generate_user_session(
        self,
        user_id: Optional[str] = None,
        event_count: int = 5
    ) -> Tuple[str, List[CookieEvent]]:
        """
        Generate a complete user session with multiple events.
        
        Args:
            user_id: Optional user ID. If None, a new one is generated
            event_count: Number of events in the session. Must be positive
            
        Returns:
            Tuple of (user_id, list of CookieEvent)
            
        Raises:
            DataGenerationError: If event_count is invalid or generation fails
        """
        if event_count <= 0:
            raise DataGenerationError("Event count must be a positive integer")
        
        try:
            user_id = user_id or self.generate_user_id()
            session_id = self.generate_session_id()
            
            session_start = datetime.utcnow()
            events = []
            
            for i in range(event_count):
                # Distribute events throughout the session (0-30 minutes)
                event_timestamp = session_start + timedelta(
                    seconds=random.randint(0, 1800)
                )
                
                event = self.generate_single_event(
                    user_id=user_id,
                    session_id=session_id,
                    timestamp=event_timestamp
                )
                events.append(event)
            
            logger.info(f"Generated session {session_id} with {event_count} events")
            return user_id, events
        except Exception as e:
            raise DataGenerationError(f"Failed to generate user session: {str(e)}")
    
    def generate_dataset(
        self,
        user_count: int = 10,
        events_per_user: int = 10
    ) -> Dict[str, List[CookieEvent]]:
        """
        Generate a complete dataset with multiple users and their events.
        
        Args:
            user_count: Number of unique users. Must be positive
            events_per_user: Events per user. Must be positive
            
        Returns:
            Dictionary mapping user_id to list of CookieEvent
            
        Raises:
            DataGenerationError: If parameters are invalid or generation fails
        """
        if user_count <= 0 or events_per_user <= 0:
            raise DataGenerationError(
                "user_count and events_per_user must be positive integers"
            )
        
        total_events = user_count * events_per_user
        if total_events > 100000:
            logger.warning(f"Generating large dataset: {total_events} total events")
        
        try:
            dataset = {}
            
            for _ in range(user_count):
                user_id, events = self.generate_user_session(
                    event_count=events_per_user
                )
                dataset[user_id] = events
            
            logger.info(
                f"Generated dataset with {user_count} users "
                f"and {total_events} total events"
            )
            return dataset
        except Exception as e:
            raise DataGenerationError(f"Failed to generate dataset: {str(e)}")
    
    def events_to_dict(self, events: List[CookieEvent]) -> List[Dict[str, Any]]:
        """
        Convert CookieEvent objects to dictionaries for serialization.
        
        Args:
            events: List of CookieEvent objects
            
        Returns:
            List of dictionaries representing events
        """
        return [
            {
                **asdict(event),
                'timestamp': event.timestamp.isoformat()
            }
            for event in events
        ]


def main() -> None:
    """
    Main function demonstrating DataGenerator usage.
    """
    try:
        logger.info("Starting data generation demonstration")
        
        # Initialize generator
        generator = DataGenerator(seed=42)
        
        # Generate a single event
        single_event = generator.generate_single_event()
        logger.info(f"Single event generated: {single_event.event_id}")
        
        # Generate multiple events
        events = generator.generate_events(count=5)
        logger.info(f"Generated {len(events)} events")
        
        # Generate user session
        user_id, session_events = generator.generate_user_session(event_count=10)
        logger.info(f"Generated session for user {user_id} with {len(session_events)} events")
        
        # Generate full dataset
        dataset = generator.generate_dataset(user_count=3, events_per_user=5)
        logger.info(f"Generated dataset with {len(dataset)} users")
        
        logger.info("Data generation demonstration completed successfully")
        
    except DataGenerationError as e:
        logger.error(f"Data generation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
