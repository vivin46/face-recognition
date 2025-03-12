from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# Load attendance data from CSV
def load_attendance_data():
    try:
        df = pd.read_csv("data/attendance.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["Name", "Timestamp"])

# Route for the dashboard
@app.route("/")
def dashboard():
    # Load attendance data
    df = load_attendance_data()

    # Convert Timestamp to datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Group by name and count attendance
    attendance_count = df["Name"].value_counts().reset_index()
    attendance_count.columns = ["Name", "Attendance Count"]

    # Create a bar chart using Plotly
    fig = px.bar(
        attendance_count,
        x="Name",
        y="Attendance Count",
        title="Attendance Count by Student",
        labels={"Name": "Student Name", "Attendance Count": "Number of Attendances"},
    )

    # Convert the Plotly chart to HTML
    chart_html = pio.to_html(fig, full_html=False)

    # Render the dashboard template with the chart and data
    return render_template("dashboard.html", chart_html=chart_html, table_data=df.to_dict("records"))

if __name__ == "__main__":
    app.run(debug=True)