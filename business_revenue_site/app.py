import os
from flask import Flask, render_template, request
from openai import OpenAI

app = Flask(__name__)

def clamp(value, low=0, high=10):
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return 0.0

def calculate_revenue_factor_score(data):
    # Inputs are on a 0-10 scale
    visibility = clamp(data.get("visibility"))
    accessibility = clamp(data.get("accessibility"))
    parking = clamp(data.get("parking"))
    footfall = clamp(data.get("footfall"))
    digital_presence = clamp(data.get("digital_presence"))
    years_of_operation = clamp(data.get("years_of_operation"))
    competition = clamp(data.get("competition"))
    construction_disruption = clamp(data.get("construction_disruption"))
    traffic_congestion = clamp(data.get("traffic_congestion"))
    rent_pressure = clamp(data.get("rent_pressure"))

    positive_score = (
        visibility * 0.22 +
        accessibility * 0.18 +
        parking * 0.12 +
        footfall * 0.22 +
        digital_presence * 0.16 +
        min(years_of_operation, 10) * 0.10
    )

    negative_score = (
        competition * 0.30 +
        construction_disruption * 0.30 +
        traffic_congestion * 0.20 +
        rent_pressure * 0.20
    )

    # Convert to 0-100 scale
    raw = (positive_score / 10.0) * 100 - (negative_score / 10.0) * 35
    score = max(0, min(100, round(raw, 2)))

    if score >= 75:
        label = "High revenue potential"
    elif score >= 50:
        label = "Moderate revenue potential"
    else:
        label = "Low revenue potential"

    return score, label

def groq_explain(score, label, form_data):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "Set GROQ_API_KEY to enable AI explanation. "
            "The local score is still working."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    prompt = f"""
    You are analyzing business revenue factors for a local business location.

    Local scoring result:
    - Revenue factor score: {score}/100
    - Category: {label}

    Input factors (0-10 scale):
    - Visibility: {form_data.get('visibility')}
    - Accessibility: {form_data.get('accessibility')}
    - Parking: {form_data.get('parking')}
    - Footfall: {form_data.get('footfall')}
    - Digital presence: {form_data.get('digital_presence')}
    - Years of operation: {form_data.get('years_of_operation')}
    - Competition pressure: {form_data.get('competition')}
    - Construction disruption: {form_data.get('construction_disruption')}
    - Traffic congestion: {form_data.get('traffic_congestion')}
    - Rent pressure: {form_data.get('rent_pressure')}

    Give:
    1. A short interpretation of the score
    2. The top 3 strongest factors
    3. The top 3 risks
    4. 3 practical actions to improve revenue
    Keep it concise and useful.
    """

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a business analytics assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return completion.choices[0].message.content.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    ai_summary = None
    form_values = {
        "visibility": 7,
        "accessibility": 7,
        "parking": 6,
        "footfall": 7,
        "digital_presence": 6,
        "years_of_operation": 5,
        "competition": 4,
        "construction_disruption": 3,
        "traffic_congestion": 4,
        "rent_pressure": 4,
    }

    if request.method == "POST":
        for key in form_values:
            form_values[key] = request.form.get(key, form_values[key])

        score, label = calculate_revenue_factor_score(form_values)
        result = {"score": score, "label": label}
        ai_summary = groq_explain(score, label, form_values)

    return render_template("index.html", result=result, ai_summary=ai_summary, form_values=form_values)

if __name__ == "__main__":
    app.run(debug=True)
