#pip install anthropic pandas thefuzz

import time
import json
import pandas as pd
import pandas.errors
import anthropic
from thefuzz import fuzz

# --- 1. CONFIGURATION ---

# *** Your file and column names ***
INPUT_FILE = "PUBMED_cohort.csv"
OUTPUT_FILE = "PUBMED_cohort_FINAL_results_Claude.csv"

# This is the column with the symptom descriptions
SYMPTOM_COLUMN = "Symptoms"
# This is your "ground truth" column
TRUTH_COLUMN = "OMIM Diagnosis"

# --- 2. SET UP CLAUDE CLIENT ---
# embedded the key as you requested.
# WARNING: Do not share this file with anyone.
try:
    client = anthropic.Anthropic(api_key="XXX")  # Replace with your actual Claude API key
    print("Claude client initialized.")
except Exception as e:
    print(f"Could not initialize Claude client. Check your API key. Error: {e}")
    # Exit if the key is wrong
    exit()

# --- 3. API FUNCTION (WITH JSON MODE & COUNT) ---
def get_diagnoses_from_claude(symptom_text, count=10):
    """
    Sends symptom text to Claude and requests a JSON list of diagnoses.
    """
    system_prompt = (
        "You are a senior clinical geneticist and neurologist expert "
        "specializing in genetic epilepsy. Your task is to provide a "
        "differential diagnosis based on clinical features, referencing OMIM."
    )

    user_prompt = f"""
    Based on the following clinical symptoms:
    "{symptom_text}"

    As a neurologist and geneticist expert, what are the most likely {count} diagnoses for genetic epilepsy?
    Base your answer on the OMIM database.

    Please provide your answer *only* as a JSON object with a single key
    "diagnoses", which contains a list of {count} diagnosis strings.
    """

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            temperature=0.2,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Extract JSON from response
        response_text = response.content[0].text.strip()
        
        # Try to find JSON in the response
        if response_text.startswith('```json'):
            # Remove markdown code block formatting
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            # Remove generic code block formatting
            response_text = response_text[3:-3].strip()
        
        result_json = json.loads(response_text)
        return result_json.get("diagnoses", [])  # Return the list

    except json.JSONDecodeError as e:
        print(f"  !! JSON decode error for '{symptom_text[:40]}...': {e}")
        print(f"  Raw response: {response_text[:200]}...")
        return []
    except Exception as e:
        print(f"  !! Error calling Claude API for '{symptom_text[:40]}...': {e}")
        # This will return an empty list, which find_fuzzy_match can handle
        return []


# --- 4. FUZZY MATCHING FUNCTION ---

def find_fuzzy_match(api_diagnoses_list, truth_diagnosis_string):
    """
    Performs a "fuzzy match" to see if any API diagnosis is
    highly similar to the ground truth.

    Returns a tuple: (matching_string, best_score, match_found_bool)
    """

    # --- Set your sensitivity ---
    SIMILARITY_THRESHOLD = 90

    if not api_diagnoses_list or pd.isna(truth_diagnosis_string):
        return None, 0, False

    # Split the truth string by ';' to handle multiple aliases
    truth_parts = [part.strip() for part in str(truth_diagnosis_string).split(';')]

    best_match_score = 0
    best_match_string = None

    for api_dx in api_diagnoses_list:
        for truth_part in truth_parts:
            score = fuzz.token_set_ratio(api_dx, truth_part)
            if score > best_match_score:
                best_match_score = score
                best_match_string = api_dx

    # Check if the best score meets the threshold
    if best_match_score >= SIMILARITY_THRESHOLD:
        print(f"  -> Match FOUND (Score: {best_match_score}): {best_match_string}")
        return best_match_string, best_match_score, True
    else:
        print(f"  -> No match found (Best score: {best_match_score})")
        return None, best_match_score, False


# --- 5. MAIN SCRIPT ---

def main():
    """
    Main function to load, process via API, match, and save.
    """
    print(f"Loading data from '{INPUT_FILE}'...")

    # --- Robust loading logic ---
    df = None
    try:
        print("Attempting to load as standard comma-separated (CSV)...")
        df = pd.read_csv(INPUT_FILE)

        if SYMPTOM_COLUMN not in df.columns:
            print(f"  Warning: Loaded CSV, but '{SYMPTOM_COLUMN}' not found.")
            if len(df.columns) == 1 and ',' in df.columns[0]:
                print("  Detected incorrect delimiter.")
                raise pd.errors.ParserError("Incorrect delimiter, will try tab-separated.")

    except (pd.errors.ParserError, UnicodeDecodeError):
        print("Could not read as standard CSV. Attempting to read as tab-separated (TSV)...")
        try:
            df = pd.read_csv(INPUT_FILE, delimiter='\t')
        except Exception as e2:
            print(f"Could not read as TSV. Error: {e2}. Trying Excel...")
            try:
                df = pd.read_excel(INPUT_FILE)
            except Exception as e3:
                print(f"Fatal Error: Could not load file. Final error: {e3}")
                return

    except FileNotFoundError:
        print(f"Error: The file '{INPUT_FILE}' was not found.")
        return
    # --- End of loading logic ---

    # Check for required columns
    if SYMPTOM_COLUMN not in df.columns or TRUTH_COLUMN not in df.columns:
        print(f"Error: Missing required columns.")
        print(f"Need '{SYMPTOM_COLUMN}' and '{TRUTH_COLUMN}'.")
        print(f"Found columns: {df.columns.to_list()}")
        return

    print(f"Successfully loaded data. Found {len(df)} rows to process.")

    # Create lists to store all new data
    results_top_10 = []
    results_top_20 = []
    matches_top_10 = []
    matches_top_20 = []

    # Loop through each row
    for index, row in df.iterrows():
        symptom_data = str(row[SYMPTOM_COLUMN])
        truth_data = str(row[TRUTH_COLUMN])

        print(f"\nProcessing row {index + 1}/{len(df)}...")

        if pd.isna(symptom_data) or not symptom_data.strip():
            print("  Skipping row (empty symptom data).")
            results_top_10.append([])
            results_top_20.append([])
            matches_top_10.append((None, 0, False))
            matches_top_20.append((None, 0, False))
            continue

        # --- 1. Get Top 10 Diagnoses ---
        print("  Calling Claude API for Top 10...")
        api_dx_list_10 = get_diagnoses_from_claude(symptom_data, count=10)
        results_top_10.append(api_dx_list_10)
        # Find match in Top 10
        match_10_tuple = find_fuzzy_match(api_dx_list_10, truth_data)
        matches_top_10.append(match_10_tuple)

        # --- Rate Limit Delay ---
        # Claude has different rate limits, using 1 second to be safe
        time.sleep(1)

        # --- 2. Get Top 20 Diagnoses ---
        print("  Calling Claude API for Top 20...")
        api_dx_list_20 = get_diagnoses_from_claude(symptom_data, count=20)
        results_top_20.append(api_dx_list_20)
        # Find match in Top 20
        match_20_tuple = find_fuzzy_match(api_dx_list_20, truth_data)
        matches_top_20.append(match_20_tuple)

        # Second delay to be safe
        time.sleep(1)

    # --- 6. SAVE RESULTS & CALCULATE PERCENTAGE ---

    print("\nProcessing complete. Saving results...")

    # Add new columns to the DataFrame
    df['Claude_Top_10_Diagnoses'] = results_top_10
    df['Claude_Top_20_Diagnoses'] = results_top_20

    # Unpack the tuples from the match results
    df['Top_10_Match_String'] = [t[0] for t in matches_top_10]
    df['Top_10_Match_Score'] = [t[1] for t in matches_top_10]
    df['Top_10_Match_Found'] = [t[2] for t in matches_top_10]

    df['Top_20_Match_String'] = [t[0] for t in matches_top_20]
    df['Top_20_Match_Score'] = [t[1] for t in matches_top_20]
    df['Top_20_Match_Found'] = [t[2] for t in matches_top_20]

    # Save to new file
    try:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Success! Results saved to '{OUTPUT_FILE}'.")
    except Exception as e:
        print(f"Error saving file: {e}")

    # Calculate final percentages
    total_rows = len(df)
    matches_10 = df['Top_10_Match_Found'].sum()
    matches_20 = df['Top_20_Match_Found'].sum()

    print(f"\n=== FINAL RESULTS ===")
    print(f"Total cases processed: {total_rows}")
    print(f"Top 10 matches: {matches_10}/{total_rows} ({matches_10/total_rows*100:.1f}%)")
    print(f"Top 20 matches: {matches_20}/{total_rows} ({matches_20/total_rows*100:.1f}%)")


if __name__ == "__main__":
    main()
