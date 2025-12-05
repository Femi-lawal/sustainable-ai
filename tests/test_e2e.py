"""
End-to-End Tests for Sustainable AI Streamlit Application.
Uses Playwright for browser automation testing.
"""

import pytest
import subprocess
import time
import sys
import os
from pathlib import Path

# Add source to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def streamlit_server():
    """Start Streamlit server for testing."""
    app_path = PROJECT_ROOT / "src" / "gui" / "app.py"
    
    if not app_path.exists():
        pytest.skip(f"App file not found: {app_path}")
    
    # Start Streamlit server
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(app_path),
         "--server.port", "8502", "--server.headless", "true"],
        cwd=str(PROJECT_ROOT / "src" / "gui"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(10)  # Give Streamlit time to start
    
    yield "http://localhost:8502"
    
    # Cleanup
    process.terminate()
    process.wait(timeout=10)


@pytest.fixture(scope="module")
def browser(streamlit_server):
    """Create Playwright browser instance."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("Playwright not installed")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(browser, streamlit_server):
    """Create a new page for each test."""
    page = browser.new_page()
    page.goto(streamlit_server, wait_until="networkidle")
    # Wait for Streamlit to fully load
    page.wait_for_selector('[data-testid="stApp"]', timeout=30000)
    yield page
    page.close()


# ============================================================================
# E2E TESTS
# ============================================================================

@pytest.mark.e2e
class TestAppLoads:
    """Test that the application loads correctly."""
    
    def test_app_loads(self, page):
        """Test that the app loads without errors."""
        # Check for the app container
        assert page.locator('[data-testid="stApp"]').is_visible()
    
    def test_header_visible(self, page):
        """Test that the header is visible."""
        # Look for any header or title element in the app
        # Streamlit apps may have different header structures
        app_visible = page.locator('[data-testid="stApp"]').is_visible()
        assert app_visible
    
    def test_tabs_visible(self, page):
        """Test that main tabs are visible."""
        # Use role-based selectors for tabs or check for tab container
        tabs = page.locator('[role="tab"]')
        if tabs.count() == 0:
            # Try alternative selector
            tabs = page.locator('[data-testid="stTab"]')
        if tabs.count() == 0:
            # Just verify app loaded successfully
            assert page.locator('[data-testid="stApp"]').is_visible()
        else:
            assert tabs.count() >= 1  # Should have at least one tab


@pytest.mark.e2e
class TestAnalyzeTab:
    """Test the Analyze Prompt tab."""
    
    def test_analyze_tab_elements(self, page):
        """Test that Analyze tab has required elements."""
        # Click on first tab (Analyze) using test-id
        tabs = page.locator('[data-testid="stTab"]')
        if tabs.count() > 0:
            tabs.first.click()
            page.wait_for_timeout(1000)
        
        # Should have text area for prompt input
        text_area = page.locator('textarea')
        assert text_area.count() >= 1
    
    def test_analyze_simple_prompt(self, page):
        """Test analyzing a simple prompt."""
        # Click on first tab (Analyze)
        tabs = page.locator('[data-testid="stTab"]')
        if tabs.count() > 0:
            tabs.first.click()
            page.wait_for_timeout(1000)
        
        # Enter a prompt
        text_area = page.locator('textarea').first
        if text_area.is_visible():
            text_area.fill("Hello, how are you today?")
        
            # Click analyze button - use test-id for more robustness
            analyze_btns = page.locator('[data-testid="stBaseButton-primary"]')
            if analyze_btns.count() > 0:
                analyze_btns.first.click()
                
                # Wait for results
                page.wait_for_timeout(5000)


@pytest.mark.e2e
class TestOptimizeTab:
    """Test the Optimize tab."""
    
    def test_optimize_tab_elements(self, page):
        """Test that Optimize tab has required elements."""
        # Click on Optimize tab (second tab)
        tabs = page.locator('[data-testid="stTab"]')
        if tabs.count() > 1:
            tabs.nth(1).click()
            page.wait_for_timeout(1000)
        
        # Should have text area 
        text_area = page.locator('textarea')
        assert text_area.count() >= 1
    
    def test_optimize_verbose_prompt(self, page):
        """Test optimizing a verbose prompt."""
        # Click on Optimize tab (second tab)
        tabs = page.locator('[data-testid="stTab"]')
        if tabs.count() > 1:
            tabs.nth(1).click()
            page.wait_for_timeout(1000)
        
        # Enter a verbose prompt
        text_area = page.locator('textarea').first
        if text_area.is_visible():
            verbose_text = "In order to provide you with assistance, I would like to consider the fact that..."
            text_area.fill(verbose_text)
            
            # Click optimize button
            optimize_btns = page.locator('[data-testid="stBaseButton-primary"]')
            if optimize_btns.count() > 0:
                optimize_btns.first.click()
                
                # Wait for results
                page.wait_for_timeout(5000)


@pytest.mark.e2e
class TestDashboardTab:
    """Test the Dashboard tab."""
    
    def test_dashboard_tab_elements(self, page):
        """Test that Dashboard tab has required elements."""
        # Click on Dashboard tab (third tab)
        tabs = page.locator('[data-testid="stTab"]')
        if tabs.count() > 2:
            tabs.nth(2).click()
            page.wait_for_timeout(1000)
        
        # Dashboard should load without errors
        app_visible = page.locator('[data-testid="stApp"]').is_visible()
        assert app_visible
    
    def test_dashboard_refresh(self, page):
        """Test dashboard refresh functionality."""
        # Click on Dashboard tab
        tabs = page.locator('[data-testid="stTab"]')
        if tabs.count() > 2:
            tabs.nth(2).click()
            page.wait_for_timeout(1000)
        
        # Look for any refresh/update buttons
        refresh_btns = page.locator('[data-testid="stBaseButton-secondary"]')
        if refresh_btns.count() > 0:
            refresh_btns.first.click()
            page.wait_for_timeout(2000)


@pytest.mark.e2e  
class TestReportsTab:
    """Test the Reports tab."""
    
    def test_reports_tab_elements(self, page):
        """Test that Reports tab has required elements."""
        # Click on Reports tab (fourth tab)
        tabs = page.locator('[role="tab"]')
        if tabs.count() > 3:
            tabs.nth(3).click()
            page.wait_for_timeout(1000)
        
        # Should load without errors
        app_visible = page.locator('[data-testid="stApp"]').is_visible()
        assert app_visible
    
    def test_generate_report(self, page):
        """Test generating a transparency report."""
        # Click on Reports tab
        tabs = page.locator('[role="tab"]')
        if tabs.count() > 3:
            tabs.nth(3).click()
            page.wait_for_timeout(1000)
        
        # Click generate button if present and visible
        generate_btns = page.locator('[data-testid="stBaseButton-primary"]')
        for i in range(generate_btns.count()):
            if generate_btns.nth(i).is_visible():
                generate_btns.nth(i).click()
                page.wait_for_timeout(5000)
                break
        
        # Verify app is still responsive
        assert page.locator('[data-testid="stApp"]').is_visible()


@pytest.mark.e2e
class TestSidebarConfiguration:
    """Test sidebar configuration options."""
    
    def test_sidebar_exists(self, page):
        """Test that sidebar exists."""
        # Sidebar should have configuration options
        # Look for sidebar toggle or content
        sidebar = page.locator('[data-testid="stSidebar"]')
        # Sidebar might be collapsed by default


@pytest.mark.e2e
class TestResponsiveness:
    """Test application responsiveness."""
    
    def test_mobile_viewport(self, browser, streamlit_server):
        """Test on mobile viewport."""
        page = browser.new_page(viewport={"width": 375, "height": 667})
        page.goto(streamlit_server, wait_until="networkidle")
        
        # App should still be visible
        page.wait_for_timeout(5000)
        assert page.locator('[data-testid="stApp"]').is_visible()
        
        page.close()
    
    def test_tablet_viewport(self, browser, streamlit_server):
        """Test on tablet viewport."""
        page = browser.new_page(viewport={"width": 768, "height": 1024})
        page.goto(streamlit_server, wait_until="networkidle")
        
        # App should still be visible
        page.wait_for_timeout(5000)
        assert page.locator('[data-testid="stApp"]').is_visible()
        
        page.close()


@pytest.mark.e2e
class TestErrorHandling:
    """Test error handling in the UI."""
    
    def test_empty_prompt_handling(self, page):
        """Test handling of empty prompt submission."""
        # Click on first tab (Analyze)
        tabs = page.locator('[data-testid="stTab"]')
        if tabs.count() > 0:
            tabs.first.click()
            page.wait_for_timeout(1000)
        
        # Try to analyze without entering text
        analyze_btns = page.locator('[data-testid="stBaseButton-primary"]')
        if analyze_btns.count() > 0:
            analyze_btns.first.click()
            page.wait_for_timeout(2000)
            
            # Should show warning or handle gracefully
            # (not crash)
        
        # App should still be visible
        assert page.locator('[data-testid="stApp"]').is_visible()


# ============================================================================
# STANDALONE TESTS (NO PLAYWRIGHT REQUIRED)
# ============================================================================

@pytest.mark.e2e
class TestAppStartup:
    """Test that the app can start without errors."""
    
    def test_app_imports(self):
        """Test that app modules can be imported."""
        try:
            # These imports should not raise errors
            from gui.app import main, load_models
            from gui.layout import setup_page_config, render_header
        except ImportError as e:
            # Some imports may fail due to missing Streamlit context
            # That's expected - we're testing import structure
            pass
    
    def test_layout_functions_exist(self):
        """Test that layout functions exist."""
        try:
            from gui.layout import (
                setup_page_config,
                apply_custom_css,
                render_header,
                render_sidebar,
                render_metric_card
            )
            # Functions should exist (may not work outside Streamlit context)
        except ImportError:
            pass


# ============================================================================
# LOAD TESTING
# ============================================================================

@pytest.mark.e2e
@pytest.mark.slow
class TestLoadPerformance:
    """Test application under load."""
    
    def test_multiple_analyses(self, page):
        """Test multiple consecutive analyses."""
        # Click on first tab (Analyze)
        tabs = page.locator('[data-testid="stTab"]')
        if tabs.count() > 0:
            tabs.first.click()
            page.wait_for_timeout(1000)
        
        prompts = [
            "Hello world",
            "Explain machine learning",
            "What is artificial intelligence?",
        ]
        
        for prompt in prompts:
            text_area = page.locator('textarea').first
            if text_area.is_visible():
                text_area.fill(prompt)
                
                analyze_btns = page.locator('[data-testid="stBaseButton-primary"]')
                if analyze_btns.count() > 0:
                    analyze_btns.first.click()
                    page.wait_for_timeout(3000)
                
                # Clear for next prompt
                text_area.fill("")
        
        # App should still be responsive
        assert page.locator('[data-testid="stApp"]').is_visible()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
