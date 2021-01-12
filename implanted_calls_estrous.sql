'''
SQL query: implanted calls by estrous state
(results in 6343 unique calls)
'''

select distinct c.usv_id, c.rec_id, c.caller, c.caller_sex, r.session_id, s.subject, s.subject_status from calls c

inner join recordings r

on c.rec_id = r.rec_id

inner join sessions s

on r.session_id = s.session_id and c.caller = s.subject where c.caller_sex = 'female'

order by c.usv_id;
