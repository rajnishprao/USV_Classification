


CREATE TABLE Subjects (
name VARCHAR(10) PRIMARY KEY,
sex VARCHAR(10)
);

CREATE TABLE Stimuli (
name VARCHAR(10) PRIMARY KEY,
sex VARCHAR(10)
);

CREATE TABLE Objects (
name VARCHAR(10) PRIMARY KEY
)

CREATE TABLE Sessions (
session_id INT PRIMARY KEY,
session_date DATE,
subject VARCHAR(10) REFERENCES Subjects (name) ON DELETE CASCADE,
subject_status VARCHAR(10),
rec_count INT
);

CREATE TABLE Recordings (
rec_id INT PRIMARY KEY,
session_id INT REFERENCES Sessions (session_id) ON DELETE CASCADE,
paradigm VARCHAR(10),
stimulus VARCHAR(10) REFERENCES Stimuli (name) ON DELETE CASCADE,
stimulus_status VARCHAR(10),
stimulus_side VARCHAR(10),
stimulus_novelty BIT,
objects VARCHAR(10) REFERENCES Objects (name) ON DELETE CASCADE,
object_side VARCHAR(10),
object_novelty BIT,
start_time NUMERIC,
end_time NUMERIC
);

CREATE TABLE Events (
id SERIAL PRIMARY KEY,
rec_id INT REFERENCES Recordings (rec_id) ON DELETE CASCADE,
event_type VARCHAR(25),
start_time NUMERIC,
end_time NUMERIC,
stimulus VARCHAR(10) REFERENCES Stimuli (name) ON DELETE CASCADE,
objects VARCHAR(10) REFERENCES Objects (name) ON DELETE CASCADE
);

CREATE TABLE CallTypes (
calltype_id INT PRIMARY KEY,
abbr VARCHAR(10),
calltype VARCHAR(20),
description VARCHAR(50),
modulation_type VARCHAR(10),
category VARCHAR(10),
color VARCHAR(10)
);


CREATE TABLE Calls (
id SERIAL PRIMARY KEY,
rec_id INT REFERENCES Recordings (rec_id) ON DELETE CASCADE,
calltype_id INT REFERENCES CallTypes (calltype_id) ON DELETE CASCADE,
Nlx_time NUMERIC,
Nlx_adjusted NUMERIC,
duration NUMERIC,
contact BIT,
caller VARCHAR(10),
caller_sex VARCHAR(10)
);



-------------------




CREATE TABLE STC (
id SERIAL PRIMARY KEY,
session_id VARCHAR(10),
tetrode_id VARCHAR(10),
rel_depth VARCHAR(10),
layer VARCHAR(10),
brain_structure DATE,
relative_position VARCHAR(10),
receptive_field VARCHAR(10),
histo VARCHAR(10),
barrel_septum VARCHAR(10),
bregma DATE,
comment VARCHAR(10),
cluster_id VARCHAR(10),
rel_depth VARCHAR(10),
layer VARCHAR(10),
brain_structure DATE,
id SERIAL PRIMARY KEY,
session_id VARCHAR(10),
tetrode_id VARCHAR(10),
rel_depth VARCHAR(10),
layer VARCHAR(10),
brain_structure DATE,
FOREIGN KEY ??
  REFERENCES ?? ON DELETE CASCADE
);

CREATE TABLE Rats (
id SERIAL PRIMARY KEY,
purpose VARCHAR(10),
name VARCHAR(10),
sex VARCHAR(10),
pack VARCHAR(10),
birth_date DATE,
FOREIGN KEY ??
  REFERENCES ?? ON DELETE CASCADE
);
