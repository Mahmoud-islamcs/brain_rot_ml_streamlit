import os
import time
# pyrefly: ignore [missing-import]
from playwright.sync_api import sync_playwright

# Configuration Constants
APP_URL = "https://brainrot-analytics-egy.streamlit.app/"
OUTPUT_DIR = "screenshots"
VIEWPORT_SIZE = {"width": 1920, "height": 1080}

NAVIGATION_ITEMS = [
    ("Predict", "01_predict_page.png"),
    ("Batch Prediction", "02_batch_prediction_page.png"),
    ("Dataset Explorer", "03_dataset_explorer_page.png"),
    ("Insights", "04_insights_page.png"),
    ("Geospatial Analysis", "05_geospatial_analysis_page.png"),
    ("About", "06_about_page.png"),
]


def navigate_to_sidebar_item(page, label: str):
    """Locates and clicks a sidebar navigation item using flexible Streamlit locators."""
    sidebar = page.locator('[data-testid="stSidebar"]')

    # 1. Try exact match inside Streamlit sidebar container
    target = sidebar.get_by_text(label, exact=True)

    # 2. Substring match inside Streamlit sidebar container
    if target.count() == 0:
        target = sidebar.get_by_text(label)

    # 3. Fallback: Exact match across entire page DOM
    if target.count() == 0:
        target = page.get_by_text(label, exact=True)

    # 4. Fallback: Substring match across entire page DOM
    if target.count() == 0:
        target = page.get_by_text(label)

    if target.count() > 0:
        target.first.click(timeout=10000)
        return True
    else:
        raise RuntimeError(f"Could not locate sidebar navigation element for: {label}")


def run_qa_suite():
    """Executes automated Playwright visual testing suite across all Streamlit application pages."""
    # Ensure output screenshots directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Target Directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Target Deployed URL: {APP_URL}")
    print("Launching Chromium browser in 1920x1080 Full HD viewport...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport=VIEWPORT_SIZE)
        page = context.new_page()

        try:
            print("Navigating to application entry point with 60000ms timeout...")
            page.goto(APP_URL, timeout=60000)

            # Wait for Streamlit app hydration and initial load
            time.sleep(6)

            for label, filename in NAVIGATION_ITEMS:
                print(f"Processing navigation target: {label}")
                save_path = os.path.join(OUTPUT_DIR, filename)

                try:
                    print(f"Clicking navigation item: {label}")
                    navigate_to_sidebar_item(page, label)

                    # Sleep buffer for Streamlit rerendering, Plotly charts, and CSS elements
                    time.sleep(5)

                    # Wait for Plotly charts if present on page
                    try:
                        page.wait_for_selector(".js-plotly-plot", timeout=3000)
                        time.sleep(1)
                    except Exception:
                        pass

                    page.screenshot(path=save_path, full_page=True)
                    print(f"Successfully captured screenshot for '{label}' at: {save_path}")

                except Exception as nav_err:
                    print(f"Error during navigation or rendering for '{label}': {str(nav_err)}")
                    print(f"Attempting fallback screenshot of current view for '{label}'...")
                    try:
                        page.screenshot(path=save_path, full_page=True)
                        print(f"Fallback screenshot saved: {save_path}")
                    except Exception as fallback_err:
                        print(f"Failed to capture fallback screenshot for '{label}': {str(fallback_err)}")

            print("Automation suite execution completed across all navigation items.")

        except Exception as suite_err:
            print(f"Global suite execution encountered an error: {str(suite_err)}")

        finally:
            context.close()
            browser.close()
            print("Browser context closed cleanly.")


if __name__ == "__main__":
    run_qa_suite()
