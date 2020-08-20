CREATE TABLE Sessions (
session_id INT PRIMARY KEY,
session_date DATE,
subject VARCHAR(10) REFERENCES Subjects (name) ON DELETE CASCADE,
subject_status VARCHAR(10),
rec_count INT
);

INSERT INTO Sessions (session_id, session_date, subject, subject_status, rec_count) VALUES
(50,'2012-04-26','R8','null',6),
(60,'2012-08-07','R10','null',4),
(63,'2012-08-10','R10','null',4),
(67,'2012-08-15','R10','null',4),
(87,'2013-02-22','R13','me',5),
(98,'2013-04-17','R14','es',5),
(99,'2013-04-18','R14','di',5)
;


