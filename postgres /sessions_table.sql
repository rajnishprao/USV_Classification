CREATE TABLE Sessions (
session_id INT PRIMARY KEY,
session_date DATE,
subject VARCHAR(10) REFERENCES Subjects (name) ON DELETE CASCADE,
subject_status VARCHAR(10),
rec_count INT
);

INSERT INTO Sessions (session_id, session_date, subject, subject_status, rec_count) VALUES
(71,'2012-08-27','R11','null',6),
(72,'2012-08-28','R11','null',6),
(73,'2012-08-29','R11','null',6),
(74,'2012-08-30','R11','null',6),
(75,'2012-08-31','R11','null',6)
;


