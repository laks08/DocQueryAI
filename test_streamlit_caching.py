#!/usr/bin/env python3
"""
Test to verify Streamlit caching is working properly
"""

import streamlit as st
import time
import logging

# Configure logging to see when functions are called
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def expensive_initialization():
    """Simulate expensive initialization that should only run once"""
    logger.info("🚀 EXPENSIVE INITIALIZATION RUNNING - This should only appear once!")
    time.sleep(2)  # Simulate slow initialization
    return "Initialization Complete"

def main():
    st.title("Streamlit Caching Test")
    
    # This should only run once and be cached
    result = expensive_initialization()
    st.success(f"✅ {result}")
    
    # Button that triggers rerun
    if st.button("Click me - should NOT trigger re-initialization"):
        st.info("Button clicked! Check the logs - initialization should NOT run again.")
    
    # Show current time to prove the app is rerunning
    st.write(f"Current time: {time.time()}")
    
    st.info("""
    **How to test:**
    1. Watch the terminal logs when you first load this page
    2. You should see "EXPENSIVE INITIALIZATION RUNNING" once
    3. Click the button multiple times
    4. The initialization message should NOT appear again in logs
    5. Only the current time should change
    """)

if __name__ == "__main__":
    main()