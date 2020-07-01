CREATE TABLE Events (
id SERIAL PRIMARY KEY,
rec_id INT REFERENCES Recordings (rec_id) ON DELETE CASCADE,
event_type VARCHAR(25),
start_time NUMERIC,
end_time NUMERIC,
stimulus VARCHAR(10) REFERENCES Stimuli (name) ON DELETE CASCADE,
objects VARCHAR(10) REFERENCES Objects (name) ON DELETE CASCADE
);

