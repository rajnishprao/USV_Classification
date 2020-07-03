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

INSERT INTO Recordings (rec_id,session_id,paradigm,stimulus,stimulus_status,stimulus_side,stimulus_novelty,objects,object_side,object_novelty,start_time,end_time) VALUES
(71001,71,'partner','R11-F1','es','front','1','null','front','0',1381582.903, 2281328.511),
(71002,71,'partner','R11-M2','null','front','1','null','front','0',2322997.487, 3228626.282),
(71003,71,'partner','R11-F1','es','front','1','null','front','0',3275673.778, 4173613.387),
(71004,71,'partner','R11-M2','null','front','1','null','front','0',4261111.597, 5158844.431),
(71005,71,'partner','R11-F1','es','front','1','null','front','0',5293552.797,6190957.756),
(71006,71,'partner','R11-M2','null','front','1','null','front','0',6289611.718,7190362.761)
;
