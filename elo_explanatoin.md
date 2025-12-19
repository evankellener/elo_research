# How the elo should work

When running a GA algorithm you should start with initial parameters(ie k, denom, ko_weight, sub_weight, etc.)

you should start when the tournament style fight end(early 2000's/ late 1990's not sure the exact date)

the elo's should be calculated for all fights. HOwever when the evaluations are mand( accuracy, ROI, log loss, etc.) it should only be on fighter's who have had more than one fifhgt(ie elo isn't 1500[there could be a small chance of an anomaly happening where a fighter's elo converges back to 1500 after a bunch a fights but it's not that likely])

the elo should be calculated for all of the interleaved_cleaned.csv however it should only be evalauted on the last years worth of fights in the interleaved, ie the fitness fucntion shoudl be evaluated on the last years worth of fights not the whole elo hiistory. In addition you should calculate all the metrics ont he past3_events.csv to test whether or not the out of sample follows the same trends as the last years worth of fights. 


One last note:

There shoudl be two different elo values, precomp_elo and postcomp_elo one to signify the elo before the match and one to signify elo at the end of the match. Make sure when you're diong the evaluations it's done with the precomp_elo howevere when testsing the out of sample elo's use the postcomp_elo's the from the fighter's previous fight as the OOS fights don't have a row inthe interleaved_cleaned.csv