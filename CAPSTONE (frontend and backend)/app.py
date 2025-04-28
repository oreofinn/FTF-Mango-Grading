from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

grading_data = {}
grading_started = False
grading_stopped = False
grade_counts = {"A": 0, "B": 0, "C": 0, "Rejected": 0}

@app.route('/')
def index():
    return redirect(url_for('status'))

@app.route('/status', methods=['GET', 'POST'])
def status():
    global grading_data, grading_started
    if request.method == 'POST':
        grading_data = request.form.to_dict()
        grading_started = False
        return redirect(url_for('grading'))
    return render_template('status.html', grading_data=grading_data)

@app.route('/grading')
def grading():
    return render_template('grading.html', grading_data=grading_data, grading_started=grading_started)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@socketio.on('request_data')
def handle_request_data():
    emit('status_data', grading_data)

@socketio.on('start_grading')
def handle_start():
    global grading_started
    grading_started = True
    emit('grading_started', broadcast=True)

@socketio.on('pause_grading')
def handle_pause():
    emit('grading_paused', broadcast=True)

@socketio.on('stop_grading')
def handle_stop():
    global grading_stopped
    grading_stopped = True
    emit('grading_stopped', broadcast=True)

@socketio.on('restart_grading')
def handle_restart():
    global grading_started, grading_data, grade_counts
    if not grading_started:
        grading_data = {}
        grade_counts = {"A": 0, "B": 0, "C": 0, "Rejected": 0}
        emit('grading_restarted', broadcast=True)

@socketio.on('grade_detected')
def handle_grade_detected(data):
    grade = data.get('grade')
    if grade in grade_counts:
        grade_counts[grade] += 1
        emit('grade_update', grade_counts, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, debug=True)
