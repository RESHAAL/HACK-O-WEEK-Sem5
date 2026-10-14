from flask import Flask, jsonify

app = Flask(__name__)

students = [
    {"id": 1, "name": "Rahul", "course": "AIML"},
    {"id": 2, "name": "Priya", "course": "CSE"},
    {"id": 3, "name": "Reshal", "course": "AIML"}
]

@app.route("/students", methods=["GET"])
def get_students():
    return jsonify(students)

if __name__ == "__main__":
    app.run(debug=True)