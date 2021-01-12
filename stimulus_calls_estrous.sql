'''
SQL query: stimulus calls by estrous state
(results in 3699 unique calls)
'''

select distinct c.usv_id, c.rec_id, c.caller, c.caller_sex, r.rec_id, r.stimulus, r.stimulus_status  from calls c

inner join recordings r

on c.rec_id = r.rec_id and c.caller = r.stimulus where c.caller_sex = 'female'

order by c.usv_id;
