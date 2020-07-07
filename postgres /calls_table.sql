CREATE TABLE Calls (
id SERIAL PRIMARY KEY,
calltype_id INT REFERENCES CallTypes (calltype_id) ON DELETE CASCADE,
USV_id INT,
rec_id INT,
Nlx_time NUMERIC,
Nlx_adjusted NUMERIC,
duration NUMERIC,
contact BIT,
caller VARCHAR(10),
caller_sex VARCHAR(10)
);




