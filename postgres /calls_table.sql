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
