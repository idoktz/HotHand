import numpy as np
import scipy.stats
import scipy.interpolate
import scipy.optimize
import matplotlib
import matplotlib.pyplot as plt
import time
import NBA_Parser
import sqlite3
from itertools import islice
from datetime import datetime
import multiprocessing as mp
import os
import sys


def GenPermLog(arr: np.ndarray, n: int) -> np.ndarray:
	a = np.atleast_2d(arr)
	b = np.arange(arr.shape[0])

	for i in range(n):
		c = np.random.permutation(b)
		d = np.take(a, c, 1)
		a = np.concatenate((a, d))

	return a


# def GenPermReg(arr: np.ndarray) -> np.ndarray:
# 	a = np.atleast_2d(arr)
# 	b = np.tile(a, (8000, 1))
#
# 	for i in range(1, 8000):
# 		np.random.shuffle(b[i])
#
# 	return b
def GenPermReg(arr):
	b = [arr.copy() for _ in range(8000)]

	for i in range(1, 8000):
		np.random.shuffle(b[i])

	return b

def calcArea(a, b, scoreA: int, scoreB: int) -> float:
	m = scoreA / scoreB
	m_i = scoreB / scoreA
	last_side, cur_side = 1, 1  # 1 is Home team
	lastPoint, curPoint = [0] * 2, [0] * 2
	curSum, cur_y, last_y, x_cut = 0., 0., 0., 0.
	#  j = 1

	for i in range(min(len(a), len(b))):
		for j in range(2):
			if j == 0:
				if type(a[i]) is not int:
					continue
				curPoint[1] += a[i]
			else:
				if type(b[i]) is not int:
					continue
				curPoint[0] += b[i]

			if curPoint[0] == lastPoint[0] and curPoint[1] == lastPoint[1]:
				continue

			cur_y = curPoint[0]*m
			# dot product used to find the side: (P.y - A.y)*(B.x - A.x) - (P.x - A.x)*(B.y - A.y)
			cur_side = 1 if (curPoint[1] > cur_y) else -1

			if cur_side == 1 and last_side == 1: # new point is at A side if last_side == 1: # lsat point was at A side
				curSum += (curPoint[1] - (cur_y + last_y)/2) * (curPoint[0] - lastPoint[0]) # calculate the local area between the graphs
			elif cur_side == -1:
				if last_side == -1:
					curSum += ((cur_y + last_y)/2 - curPoint[1]) * (curPoint[0] - lastPoint[0]) # calculate the local area between the graphs
				else:
					x_cut = curPoint[1] * m_i
					curSum += (curPoint[1] - (curPoint[1] + last_y)/2) * (x_cut - lastPoint[0]) + ((cur_y + curPoint[0])/2 - curPoint[1]) * (curPoint[0] - x_cut)

			last_side, last_y, lastPoint = cur_side, cur_y, curPoint.copy()

	if len(a) > len(b) or len(b) > len(a):
		if len(a) > len(b):
			if type(a[i+1]) is not int:
				return curSum
			curPoint[1] += a[i + 1]
		else:
			if type(b[i+1]) is not int:
				return curSum
			curPoint[0] += b[i + 1]

		if curPoint[0] == lastPoint[0] and curPoint[1] == lastPoint[1]:
			return curSum

		cur_y = curPoint[0] * m
		# dot product used to find the side: (P.y - A.y)*(B.x - A.x) - (P.x - A.x)*(B.y - A.y)
		cur_side = 1 if (curPoint[1] > cur_y) else -1

		if cur_side == 1 and last_side == 1:  # new point is at A side if last_side == 1: # lsat point was at A side
			curSum += (curPoint[1] - (cur_y + last_y) / 2) * (curPoint[0] - lastPoint[0])  # calculate the local area between the graphs
		elif cur_side == -1:
			if last_side == -1:
				curSum += ((cur_y + last_y) / 2 - curPoint[1]) * (curPoint[0] - lastPoint[0])  # calculate the local area between the graphs
			else:
				x_cut = curPoint[1] * m_i
				curSum += (curPoint[1] - (curPoint[1] + last_y) / 2) * (x_cut - lastPoint[0]) + ((cur_y + curPoint[0]) / 2 - curPoint[1]) * (curPoint[0] - x_cut)

		last_side, last_y, lastPoint = cur_side, cur_y, curPoint.copy()

	return curSum


def calcArea_global(glob, scoreA: int, scoreB: int) -> float:
	m = scoreA / scoreB
	m_i = scoreB / scoreA
	last_side, cur_side = 1, 1  # 1 is Home team
	lastPoint, curPoint = [0] * 2, [0] * 2
	curSum, cur_y, last_y, x_cut = 0., 0., 0., 0.
	j = 0

	for i in range(len(glob)):
		if j == 0:
			curPoint[1] += glob[i]
		else:
			curPoint[0] += glob[i]

		if curPoint[0] == lastPoint[0] and curPoint[1] == lastPoint[1]:
			continue

		cur_y = curPoint[0]*m
		# dot product used to find the side: (P.y - A.y)*(B.x - A.x) - (P.x - A.x)*(B.y - A.y)
		cur_side = 1 if (curPoint[1] > cur_y) else -1

		if cur_side == 1 and last_side == 1: # new point is at A side if last_side == 1: # lsat point was at A side
			curSum += (curPoint[1] - (cur_y + last_y)/2) * (curPoint[0] - lastPoint[0]) # calculate the local area between the graphs
		elif cur_side == -1:
			if last_side == -1:
				curSum += ((cur_y + last_y)/2 - curPoint[1]) * (curPoint[0] - lastPoint[0]) # calculate the local area between the graphs
			else:
				x_cut = curPoint[1] * m_i
				curSum += (curPoint[1] - (curPoint[1] + last_y)/2) * (x_cut - lastPoint[0]) + ((cur_y + curPoint[0])/2 - curPoint[1]) * (curPoint[0] - x_cut)

		last_side, last_y, lastPoint = cur_side, cur_y, curPoint.copy()
		j = 1 - j

	return curSum


def calc_possessions_change(a, b):
	
	cur_direction = -1
	changes_count = 0
	
	for i in range(min(len(a), len(b))):
		for j in range(2):
			if j == 0:
				if type(a[i]) is int and a[i] > 0:
					if cur_direction == 1:
						cur_direction = 0
						changes_count += 1
					elif cur_direction == -1:
						cur_direction = 0
			else:
				if type(b[i]) is int and b[i] > 0:
					if cur_direction == 0:
						cur_direction = 1
						changes_count += 1
					elif cur_direction == -1:
						cur_direction = 1

	if len(b) > len(a):
		for i in range(len(a), len(b)):
			if type(b[i]) is int and b[i] > 0:
				if cur_direction == 0:
					cur_direction = 1
					changes_count += 1
				elif cur_direction == -1:
					cur_direction = 1

	elif len(a) > len(b):
		for i in range(len(b), len(a)):
			if type(a[i]) is int and a[i] > 0:
				if cur_direction == 1:
					cur_direction = 0
					changes_count += 1
				elif cur_direction == -1:
					cur_direction = 0

	return changes_count


def calc_max_point_distance(a, b, score_a, score_b):
	lastPoint, curPoint = [0] * 2, [0] * 2
	cur_dist, max_dist = 0., 0.

	for i in range(min(len(a), len(b))):
		for j in range(2):
			if j == 0:
				if type(a[i]) is not int:
					continue
				curPoint[1] += a[i]
			else:
				if type(b[i]) is not int:
					continue
				curPoint[0] += b[i]

			if curPoint[0] == lastPoint[0] and curPoint[1] == lastPoint[1]:
				continue

			cur_dist = abs(score_a*curPoint[0] - score_b*curPoint[1]) / (score_a**2 + score_b**2)**0.5

			if cur_dist > max_dist:
				max_dist = cur_dist

	if len(a) > len(b) and (type(a[i+1]) is int) and a[i+1] > 0:
		curPoint[1] += a[i+1]
		cur_dist = abs(score_a*curPoint[0] - score_b*curPoint[1]) / (score_a**2 + score_b**2)**0.5

		if cur_dist > max_dist:
			max_dist = cur_dist

	return max_dist


def check_res(game_tag, season, scoreA, scoreB):
	game_possessions, _ = NBA_Parser.pbp_to_possessions("Games/{}-{}/{}.csv".format(season-1, season, game_tag))
	a_poss = [_ for _ in game_possessions[0::2] if type(_) == int]
	b_poss = [_ for _ in game_possessions[1::2] if type(_) == int]
	a_perms = GenPermReg(a_poss)
	b_perms = GenPermReg(b_poss)
	res_area = [0] * 8000
	for i in range(8000):
		res_area[i] = calcArea(a_perms[i], b_perms[i], scoreA, scoreB)

	print(np.median(res_area), res_area[0])
	# print(res_area[0])


def calc_momentum(start_season, end_season, lock, prc_name):
	with sqlite3.connect('Games.db') as conn_db:
		insert_arr = [(None,) * 2] * 2000  # 2000 - max number of games per season
		insert_row = [None] * 2
		# res_area = [0.] * 8000

		for i in range(start_season, end_season - 1, -1):
			result = conn_db.execute("SELECT COUNT(*) FROM games_table WHERE SEASON=?", (i,))
			num_of_rows = result.fetchone()[0]
			games_list = conn_db.execute("SELECT GAME_TAG, SEASON, HOME_SCORE, AWAY_SCORE FROM games_table WHERE SEASON=?", (i,))
			# time_var = 0
			# EMA_var = 0
			row_num = 0

			for index, game in enumerate(games_list):
				start_t = time.time()
				for k in range(1, 6):
					try:
						try:
							game_possessions = NBA_Parser.pbp_to_possessions("Games/{}-{}/{}.csv".format(game[1] - 1, game[1], game[0]))
						except:
							raise Exception("pbp parser error")

						# if who_start == 'H':
						# 	scoreA = game[2]
						# 	scoreB = game[3]
						# else:
						# 	scoreA = game[3]
						# 	scoreB = game[2]
						#
						# a_poss = [_ for _ in game_possessions[0::2] if type(_) == int]
						# b_poss = [_ for _ in game_possessions[1::2] if type(_) == int]
						#
						# if scoreA != sum(a_poss) or scoreB != sum(b_poss):
						# 	raise Exception("Sum doesn't match!")
						#
						# a_perms = GenPermReg(a_poss)
						# b_perms = GenPermReg(b_poss)
						#
						# for j in range(8000):
						# 	res_area[j] = calcArea(a_perms[j], b_perms[j], scoreA, scoreB)
						#
						insert_row[0] = game[0]
						insert_row[1] = '|'.join([repr(_) for _ in game_possessions])
						# insert_row[2] = res_area[0]
						# insert_row[3] = np.median(res_area)
						# insert_row[4] = np.mean(res_area)
						# insert_row[5] = np.var(res_area, ddof=1)  # unbiased variance estimator
						# insert_row[6] = np.sum([(1 if (v <= insert_row[2]) else 0) for v in res_area]) / 8000

						insert_arr[row_num] = tuple(insert_row)
					except Exception as inst:
						print("\nError occurred in season {},\t{}:\t{}".format(i, game[0], inst.args))
						if row_num > 0:
							conn_db.executemany("INSERT INTO games_momentum_new VALUES (?,?)", insert_arr[0:row_num])
							conn_db.commit()
							row_num = 0
						break
					except:
						# sys.stdout.write("\r{} failed calculating,\t{} out of 5".format(game[0], k))
						# sys.stdout.flush()
						if k == 5:
							if row_num > 0:
								conn_db.executemany("INSERT INTO games_momentum_new VALUES (?,?)", insert_arr[0:row_num])
								conn_db.commit()
							# conn.close()
							raise
					else:
						row_num += 1
						if k > 1:
							print()
						break

				# sys.stdout.write("\r{}% finished ({} avg sec for game:\t{} min remains)".format(index/num_of_rows*100, time_var, time_var*(num_of_rows - index)/60))
				# sys.stdout.flush()
				if index == num_of_rows - 1:
					print("{}:\tseason {}, index {} - {}".format(prc_name, i, index, time.time() - start_t))

			if row_num > 0:
				conn_db.executemany("INSERT INTO games_momentum_new VALUES (?,?)", insert_arr[0:row_num])
				conn_db.commit()
			print("\rfinished season {}".format(i))


def csv_to_possessions():
	# with sqlite3.connect('Games.db') as conn:
		# conn.execute("""
		# CREATE TABLE IF NOT EXISTS games_momentum_new(
		# 	GAME_TAG TEXT NOT NULL UNIQUE,
		# 	POSSESSIONS_ARR TEXT NOT NULL
		# );
		# """)
	lock = mp.Lock()
	# process_arr = [None] * 2
	# process_arr[0] = mp.Process(target=calc_momentum, args=(2017, 2010, lock, 'Prc_1'))
	# process_arr[0].start()
	# process_arr[1] = mp.Process(target=calc_momentum, args=(2009, 2002, lock, 'Prc_2'))
	# process_arr[1].start()

	# process_arr[0].join()
	# process_arr[1].join()
	calc_momentum(2018, 2018, lock, 'Prc_1')


def area_hist(games_data):
	# with sqlite3.connect('Games.db') as conn:
	#
	# 	result = conn.execute("SELECT COUNT(*) FROM games_area_momentum")
	# 	num_of_rows = result.fetchone()[0]
	# 	games_stats = conn.execute("""SELECT games_table.YEAR, games_table.MONTH, games_table.HOME_SCORE, games_table.AWAY_SCORE, games_area_momentum.AREA, games_area_momentum.MEAN, games_area_momentum.VAR
	# 	FROM games_area_momentum
	# 	INNER JOIN games_table
	# 	ON games_area_momentum.GAME_TAG = games_table.GAME_TAG
	# 	""")
	#
	# 	games_data = games_stats.fetchall()
	stats_list = np.array([_[6:] for _ in games_data])
	hist_arr = (stats_list[:, 0] - stats_list[:, 2]) / np.sqrt(stats_list[:, 3])

	z_hist_density = plt.hist(hist_arr, bins=100, rwidth=0.75, density=True)[0:2]
	z_median = np.median(hist_arr)
	median_value = 0
	for i in range(len(z_hist_density[1]) - 1):
		if z_median >= z_hist_density[1][i] and z_median < z_hist_density[1][i+1]:
			median_value = z_hist_density[0][i]
	print("Median:", z_median, median_value, abs(z_median*2*np.sqrt(len(hist_arr))*median_value), sep='\t')
	plt.show(block=False)
	plt.figure()


	#region Score difference
	attribute_arr = [abs(_[4] - _[5]) for _ in games_data]
	mask_arr = [attribute_arr[_] for _ in range(len(hist_arr)) if hist_arr[_] >= 0]
	mask_arr2 = [attribute_arr[_] for _ in range(len(hist_arr)) if hist_arr[_] < 0]
	print("Mask:", np.mean(mask_arr), np.median(mask_arr), np.std(mask_arr), 1/np.sqrt(len(mask_arr)), sep="\t")
	plt.hist([mask_arr, mask_arr2], bins=100, rwidth=0.75, density=True, color=['blue', 'green'], alpha=0.7)
	# plt.hist(mask_arr2, bins=100, rwidth=0.75, density=True, color='green', alpha=0.6)
	plt.show(block=False)

	plt.figure()
	mask_arr = [hist_arr[_] for _ in range(len(hist_arr)) if attribute_arr[_] >= 20]
	mask_arr2 = [hist_arr[_] for _ in range(len(hist_arr)) if attribute_arr[_] < 20]
	print("Mann–Whitney U test:", scipy.stats.mannwhitneyu(mask_arr, mask_arr2))
	plt.hist([mask_arr, mask_arr2], bins=100, rwidth=0.75, density=True, color=['blue', 'green'], alpha=0.7)
	# plt.hist(mask_arr2, bins=100, rwidth=0.75, density=True, color='green', alpha=0.6)
	print("Mask:", np.mean(mask_arr), np.median(mask_arr), np.std(mask_arr), 1/np.sqrt(len(mask_arr)), sep="\t")
	plt.show(block=False)

	plt.figure()
	plt.boxplot([mask_arr, mask_arr2], sym='')
	plt.show()


def str2possessions(x):
	return [int(_) if _[0] != "'" else '0' for _ in str.split(x, '|')]


def calc_global_momentum_range(games_arr, proc_name, res_queue):

		area_arr = [0.] * 8000
		stats_arr = [None] * 6
		insert_arr = [(None,)*6] * len(games_arr)

		for k, game in enumerate(games_arr):
			try:
				time_1 = time.time()

				game_tag = game[0]
				game_possessions_1 = game[1]
				home_score = game[2]
				away_score = game[3]
				who_start = game[1][1]

				game_possessions_2 = [int(_) for _ in str.split(game_possessions_1, '|') if _[0] != "'"]  # only possessions
				global_possessions = [1] * (len(game_possessions_2) * 2 - 1)

				poss_perm = GenPermReg(game_possessions_2)

				for j in range(8000):
					global_possessions[0::2] = poss_perm[j]
					area_arr[j] = calcArea_global(global_possessions, home_score + away_score, len(game_possessions_2) - 1)
				# possessions_arr_a = [_ for _ in str2possessions(game[1])[0::2] if type(_) is int]
				# possessions_arr_b = [_ for _ in str2possessions(game[1])[1::2] if type(_) is int]
				# time_4 = time.time()
				# possessions_perm_a = GenPermReg(possessions_arr_a)
				# possessions_perm_b = GenPermReg(possessions_arr_b)
				# print(time.time() - time_4)
				# time_2, time_3 = 0, 0
				#
				# for j in range(8000):
				# 	time_4 = time.time()
				# 	area_arr[j] = calcArea(possessions_perm_a[j], possessions_perm_b[j], scoreH, scoreA)  # calcArea(global_possessions[0::2], global_possessions[1::2], scoreH + scoreA, len(possessions_arr) - 1)
				# 	time_3 += time.time() - time_4

				stats_arr[0] = game_tag
				stats_arr[1] = area_arr[0]
				stats_arr[2] = np.median(area_arr)
				stats_arr[3] = np.mean(area_arr)
				stats_arr[4] = np.var(area_arr, ddof=1)
				stats_arr[5] = sum([1 for v in area_arr if v <= area_arr[0]]) / 8000
				insert_arr[k] = tuple(stats_arr)

				if (k % 100) == 0:
					print("{}:\t {} - {}".format(proc_name, k, time.time() - time_1))
			except:
				print("Error in " + game[0])
				raise

		res_queue.put(insert_arr[0:(k+1)])


def calc_global_momentum():
	with sqlite3.connect('Games.db') as conn:
		conn.execute("DROP TABLE IF EXISTS games_poss_area_global")
		conn.execute("""
		CREATE TABLE IF NOT EXISTS games_poss_area_global(
			GAME_TAG TEXT NOT NULL UNIQUE,
			AREA REAL NOT NULL,
			MEDIAN REAL NOT NULL,
			MEAN REAL NOT NULL,
			VAR REAL NOT NULL,
			P_VALUE REAL NOT NULL
		);
		""")
		result = conn.execute("SELECT COUNT(*) FROM games_momentum_new")
		num_of_rows = result.fetchone()[0]
		games_stats = conn.execute("""SELECT games_momentum_new.GAME_TAG, games_momentum_new.POSSESSIONS_ARR, games_table.HOME_SCORE, games_table.AWAY_SCORE
		FROM games_momentum_new
		INNER JOIN games_table
		ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
		""")
		games_data = games_stats.fetchall()
		# area_arr = [0.] * 8000
		# sign_arr = [0.] * 8000
		#
		# for k, game in enumerate(games_data):
		#
		# 	if k < 1000:
		# 		continue
		#
		# 	time_1 = time.time()
		# 	game_tag = game[0]
		# 	scoreH, scoreA = game[2], game[3]
		# 	possessions_arr = [_ for _ in str2possessions(game[1]) if type(_) is int]
		# 	time_4 = time.time()
		# 	possessions_perm = GenPermReg(possessions_arr)
		# 	print(time.time() - time_4)
		# 	global_possessions = [1] * (len(possessions_arr) * 2 - 1)
		# 	time_2, time_3 = 0, 0
		#
		# 	for j in range(8000):
		# 		time_4 = time.time()
		# 		global_possessions[0::2] = possessions_perm[j]
		# 		time_2 += time.time() - time_4
		# 		time_4 = time.time()
		# 		area_arr[j] = calcArea_global(global_possessions, scoreH + scoreA, len(possessions_arr) - 1)  # calcArea(global_possessions[0::2], global_possessions[1::2], scoreH + scoreA, len(possessions_arr) - 1)
		# 		time_3 += time.time() - time_4
		# 	# possessions_arr_a = [_ for _ in str2possessions(game[1])[0::2] if type(_) is int]
		# 	# possessions_arr_b = [_ for _ in str2possessions(game[1])[1::2] if type(_) is int]
		# 	# time_4 = time.time()
		# 	# possessions_perm_a = GenPermReg(possessions_arr_a)
		# 	# possessions_perm_b = GenPermReg(possessions_arr_b)
		# 	# print(time.time() - time_4)
		# 	# time_2, time_3 = 0, 0
		# 	#
		# 	# for j in range(8000):
		# 	# 	time_4 = time.time()
		# 	# 	area_arr[j] = calcArea(possessions_perm_a[j], possessions_perm_b[j], scoreH, scoreA)  # calcArea(global_possessions[0::2], global_possessions[1::2], scoreH + scoreA, len(possessions_arr) - 1)
		# 	# 	time_3 += time.time() - time_4
		#
		# 	print(time_2, time_3)
		# 	sign_arr[k] = (area_arr[0] - np.mean(area_arr)) / np.std(area_arr, ddof=1)
		# 	print("\t\t", time.time() - time_1)
		#
		# 	if k == 2000:
		# 		break
		# 	print(k)
		#
		# plt.hist(sign_arr[1000:(k+1)], bins=40, rwidth=0.75)
		# plt.show()
		# print(sign_arr[0:1000])
		# print(sign_arr[1000:(k+1)])

		res_queue = mp.Queue()
		process_arr = [None] * 4
		process_arr[0] = mp.Process(target=calc_global_momentum_range, args=(games_data[0:10000], "Prc_1", res_queue))
		process_arr[0].start()
		process_arr[1] = mp.Process(target=calc_global_momentum_range, args=(games_data[10000:], "Prc_2", res_queue))
		process_arr[1].start()
		# process_arr[2] = mp.Process(target=calc_global_momentum_range, args=(10000, 14999, games_data[10000:15000].copy(), "thread_3.txt"))
		# process_arr[2].start()
		# process_arr[3] = mp.Process(target=calc_global_momentum_range, args=(15000, 25000, games_data[15000:].copy(), "thread_4.txt"))
		# process_arr[3].start()
		print("Here1")
		# process_arr[2].join()
		# process_arr[3].join()
		insert_arr = res_queue.get()
		print("Here2")
		insert_arr = insert_arr + res_queue.get()
		print("Here3")
		process_arr[0].join()
		print("check1")
		process_arr[1].join()
		print("check2")
		conn.executemany("INSERT INTO games_poss_area_global VALUES (?,?,?,?,?,?)", insert_arr)
		conn.commit()


def plot_pbp_game(pbp_arr, who_start, home_score, away_score, draw_linear=False, draw_area=False, block=False, new_figure=True, draw_legend=True):

	game_possessions = pbp_arr.split('|')  # [(int(_) if _[0] != "'" else '0') for _ in pbp_arr.split('|') if not (_[0] == "'" and _[1] != "0")]  # quarter and OT filtering

	# game_points = [[0] * (len(a_poss) + len(b_poss) + 1), [0] * (len(a_poss) + len(b_poss) + 1)]
	game_points = [[0] * len(game_possessions), [0] * len(game_possessions)]
	quarter_indices = []
	quarter_colors = ['black', 'red', 'blue', 'green', 'purple']
	quarter_legends = ['QT #' + str(_) for _ in range(1, 5)] + ['OT']
	j, k = 0, 0

	for poss in game_possessions:
		if poss[0] == "'":
			if poss in ["'1'", "'2'", "'3'", "'4'", "'1OT'"]:
				quarter_indices.append(k)
			elif poss == "'0'":
				j = 1 - j
			continue

		game_points[j][k+1] = game_points[j][k] + int(poss)
		game_points[1-j][k+1] = game_points[1-j][k]
		k += 1
		j = 1 - j
	quarter_indices.append(k)

	if new_figure:
		plt.figure()

	for i in range(len(quarter_indices) - 1):
		plt.plot(game_points[0 if who_start == 'H' else 1][quarter_indices[i]:(quarter_indices[i+1]+1)], game_points[1 if who_start == 'H' else 0][quarter_indices[i]:(quarter_indices[i+1]+1)], color=quarter_colors[i], lw=3, label=quarter_legends[i])
	# plt.plot(game_points[0 if who_start == 'H' else 1][quarter_indices[-1]:k+1], game_points[1 if who_start == 'H' else 0][quarter_indices[-1]:k+1], color=quarter_colors[i+1], lw=4)
	if draw_linear:
		plt.plot([0, home_score], [0, away_score], '--', color='black')
	if draw_area:
		plt.fill_between(game_points[0 if who_start == 'H' else 1][0:k+1], game_points[1 if who_start == 'H' else 0][0:k+1], np.array(game_points[0 if who_start == 'H' else 1][0:k+1]) * (away_score / home_score), interpolate=True, facecolor='yellow')
	plt.title("Play-By-Play Home-Away score progress")
	plt.xlabel("Home score")
	plt.ylabel("Away score")
	if draw_legend:
		plt.legend()
	plt.show(block=block)


def games_possessions_change():
	# old function
	with sqlite3.connect('Games.db') as conn:

		# conn.execute("""
		# CREATE TABLE IF NOT EXISTS games_poss_changes(
		# 	GAME_TAG TEXT NOT NULL UNIQUE,
		# 	CHANGES_COUNT INT NOT NULL,
		# 	MEDIAN REAL NOT NULL,
		# 	MEAN REAL NOT NULL,
		# 	VAR REAL NOT NULL,
		# 	P_VALUE REAL NOT NULL
		# );
		# """)
		insert_arr = [(None,) * 6] * 2000  # 2000 - max number of games per season
		insert_row = [None] * 6
		res_changes = [0] * 8000

		for i in range(2016, 2001, -1):

			result = conn.execute("SELECT COUNT(*) FROM games_table WHERE games_table.SEASON = ?", (i,))
			num_of_rows = result.fetchone()[0]
			games_stats = conn.execute("""SELECT games_momentum.GAME_TAG, games_momentum.POSSESSIONS_ARR, games_table.HOME_SCORE, games_table.AWAY_SCORE
			FROM games_momentum
			INNER JOIN games_table
			ON games_momentum.GAME_TAG = games_table.GAME_TAG
			WHERE games_table.SEASON = ?
			""", (i,))
			games_data = games_stats.fetchall()

			for k, game in enumerate(games_data):
				time_1 = time.time()
				possessions_arr_a = [_ for _ in str2possessions(game[1])[0::2] if type(_) is int]
				possessions_arr_b = [_ for _ in str2possessions(game[1])[1::2] if type(_) is int]
				possessions_perm_a = GenPermReg(possessions_arr_a)
				possessions_perm_b = GenPermReg(possessions_arr_b)
				time_2, time_3 = 0, 0

				for j in range(8000):
					time_4 = time.time()
					res_changes[j] = calc_possessions_change(possessions_perm_a[j], possessions_perm_b[j])
					time_3 += time.time() - time_4

				print(time_3)
				insert_row[0] = game[0]
				insert_row[1] = res_changes[0]
				insert_row[2] = np.median(res_changes)
				insert_row[3] = np.mean(res_changes)
				insert_row[4] = np.var(res_changes, ddof=1)
				insert_row[5] = np.sum([(1 if (v <= insert_row[1]) else 0) for v in res_changes]) / 8000
				insert_arr[k] = tuple(insert_row)

				print("\t\t", time.time() - time_1)
				print(k, res_changes[0])

			conn.executemany("INSERT INTO games_poss_changes VALUES (?,?,?,?,?,?)", insert_arr[0:(k+1)])
			conn.commit()
			print("Finished season {}".format(i))


def area_hist_poss_changes():
	with sqlite3.connect('Games.db') as conn:

		result = conn.execute("SELECT COUNT(*) FROM games_changes_momentum")
		num_of_rows = result.fetchone()[0]
		games_stats = conn.execute("""SELECT games_table.YEAR, games_table.MONTH, games_changes_momentum.CHANGES_COUNT, games_changes_momentum.MEAN, games_changes_momentum.VAR
		FROM games_changes_momentum
		INNER JOIN games_table
		ON games_changes_momentum.GAME_TAG = games_table.GAME_TAG
		""")

		games_data = games_stats.fetchall()
		month_list = [(_[0], _[1]) for _ in games_data]
		stats_list = np.array([_[2:] for _ in games_data])

		hist_arr = (stats_list[:, 0] - stats_list[:, 1]) / np.sqrt(stats_list[:, 2])
		plt.hist(hist_arr, bins=100, rwidth=0.75, density=True)
		print(np.median(hist_arr))
		print("Hist arr:\n", np.mean(hist_arr), 1/np.sqrt(len(hist_arr)), -np.mean(hist_arr)*np.sqrt(len(hist_arr)))
		plt.show()
		plt.figure()
		# plt.plot(month_list, hist_arr, 'o')


def area_hist_poss_max_dist():
	with sqlite3.connect('Games.db') as conn:

		result = conn.execute("SELECT COUNT(*) FROM games_poss_dist")
		num_of_rows = result.fetchone()[0]
		games_stats = conn.execute("""SELECT games_table.YEAR, games_table.MONTH, games_poss_dist.MAX_DIST, games_poss_dist.MEAN, games_poss_dist.VAR
		FROM games_poss_dist
		INNER JOIN games_table
		ON games_poss_dist.GAME_TAG = games_table.GAME_TAG
		""")

		games_data = games_stats.fetchall()
		month_list = [(_[0], _[1]) for _ in games_data]
		stats_list = np.array([_[2:] for _ in games_data])

		hist_arr = (stats_list[:, 0] - stats_list[:, 1]) / np.sqrt(stats_list[:, 2])
		plt.hist(hist_arr, bins=100, rwidth=0.75, density=True)
		print(np.mean(hist_arr), np.median(hist_arr), sep="\t")
		print("Hist arr:\n", np.mean(hist_arr), 1/np.sqrt(len(hist_arr)), -np.mean(hist_arr)*np.sqrt(len(hist_arr)))
		plt.show()
		plt.figure()
		# plt.plot(month_list, hist_arr, 'o')


def area_hist_poss_global():
	with sqlite3.connect('Games.db') as conn:

		result = conn.execute("SELECT COUNT(*) FROM games_poss_area_global")
		num_of_rows = result.fetchone()[0]
		games_stats = conn.execute("""SELECT games_table.YEAR, games_table.MONTH, games_poss_area_global.AREA, games_poss_area_global.MEAN, games_poss_area_global.VAR
		FROM games_poss_area_global
		INNER JOIN games_table
		ON games_poss_area_global.GAME_TAG = games_table.GAME_TAG
		""")

		games_data = games_stats.fetchall()
		month_list = [(_[0], _[1]) for _ in games_data]
		stats_list = np.array([_[2:] for _ in games_data])

		hist_arr = (stats_list[:, 0] - stats_list[:, 1]) / np.sqrt(stats_list[:, 2])
		plt.hist(hist_arr, bins=100, rwidth=0.75, density=True)
		plt.vlines(np.mean(hist_arr), 0, 0.6, color='k', linestyles='--', label='Mean')
		plt.annotate(r"${}\sigma_{{\bar{{X}}}}$".format(-4.23), xy=(np.mean(hist_arr), 0.45), xytext=(np.mean(hist_arr)+1, 0.44), arrowprops=dict(facecolor='black', shrink=0.1, width=0.2, headwidth=4), fontsize=15)
		plt.title('Area Z-Score')
		print(np.mean(hist_arr), np.median(hist_arr), sep="\t")
		print("Hist arr:\n", np.mean(hist_arr), 1/np.sqrt(len(hist_arr)), -np.mean(hist_arr)*np.sqrt(len(hist_arr)))
		plt.show()
		plt.figure()
		# plt.plot(month_list, hist_arr, 'o')


def calc_area_h_a(start_season, end_season):
		with sqlite3.connect('Games.db') as conn:
			# conn.execute("DROP TABLE IF EXISTS games_area_momentum4")
			conn.execute("""
			CREATE TABLE IF NOT EXISTS games_area_momentum5(
				GAME_TAG TEXT NOT NULL UNIQUE,
				AREA REAL NOT NULL,
				MEDIAN REAL NOT NULL,
				MEAN REAL NOT NULL,
				VAR REAL NOT NULL,
				P_VALUE REAL NOT NULL,
				MEDIAN_SMALLER INT NOT NULL,
				MEDIAN_EQUAL INT NOT NULL
			);
			""")
			insert_arr = [(None,) * 8] * 2000  # 2000 - max number of games per season
			insert_row = [None] * 8
			res_area = [0.] * 8000
			row_index = 0

			for i in range(start_season, end_season - 1, -1):
				result = conn.execute("""
				SELECT COUNT(*)
				FROM games_momentum_new
					INNER JOIN games_table ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
					inner join games_scores on games_table.GAME_TAG = games_scores.GAME_TAG
				WHERE SEASON=? AND games_scores.VALID = 1
				""", (i,))
				num_of_rows = result.fetchone()[0]
				games_stats = conn.execute("""
				SELECT games_table.GAME_TAG, games_momentum_new.POSSESSIONS_ARR, games_table.HOME_SCORE, games_table.AWAY_SCORE
				FROM games_momentum_new
					INNER JOIN games_table ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
					inner join games_scores on games_table.GAME_TAG = games_scores.GAME_TAG
				WHERE SEASON=? AND games_scores.VALID = 1
				""", (i,))

				games_data = games_stats.fetchall()
				row_index = 0

				for k, game in enumerate(games_data):
					start_t = time.time()
					game_tag = game[0]
					game_possessions_1 = game[1]
					home_score = game[2]
					away_score = game[3]
					who_start = game[1][1]

					game_possessions_2 = [(int(_) if _[0] != "'" else '0') for _ in str.split(game_possessions_1, '|') if not (_[0] == "'" and _[1] != "0")]  # quarter and OT filtering
					# a_possessions = [_ for _ in game_possessions_2[0::2] if type(_) is int]
					# b_possessions = [_ for _ in game_possessions_2[1::2] if type(_) is int]
					a_possessions = game_possessions_2[0::2]
					b_possessions = game_possessions_2[1::2]

					a_poss_perm = GenPermReg(a_possessions)
					b_poss_perm = GenPermReg(b_possessions)

					if who_start == 'H':
						for j in range(8000):
							res_area[j] = calcArea(a_poss_perm[j], b_poss_perm[j], home_score, away_score)
					else:
						for j in range(8000):
							res_area[j] = calcArea(a_poss_perm[j], b_poss_perm[j], away_score, home_score)

					insert_row[0] = game_tag
					insert_row[1] = res_area[0]
					insert_row[2] = np.median(res_area)
					insert_row[3] = np.mean(res_area)
					insert_row[4] = np.var(res_area, ddof=1)
					insert_row[5] = sum([1 for _ in res_area if _ < res_area[0]]) / 8000
					insert_row[6] = sum([1 for _ in res_area if _ < insert_row[2]])
					insert_row[7] = sum([1 for _ in res_area if _ == insert_row[2]])

					insert_arr[row_index] = tuple(insert_row)

					row_index += 1

					if k % 50 == 0:
						print("k={} - {}".format(k, time.time() - start_t))

				conn.executemany("INSERT INTO games_area_momentum5 VALUES (?,?,?,?,?,?,?,?)", insert_arr[0:row_index])
				conn.commit()
				print("finished season {}".format(i))


def calc_changes_h_a(start_season, end_season):
	with sqlite3.connect('Games.db') as conn:
		# conn.execute("DROP TABLE IF EXISTS games_changes_momentum2")
		conn.execute("""
		CREATE TABLE IF NOT EXISTS games_changes_momentum3(
			GAME_TAG TEXT NOT NULL UNIQUE,
			CHANGES_COUNT INT NOT NULL,
			MEDIAN REAL NOT NULL,
			MEAN REAL NOT NULL,
			VAR REAL NOT NULL,
			P_VALUE REAL NOT NULL,
			MEDIAN_SMALLER INT NOT NULL,
			MEDIAN_EQUAL INT NOT NULL
		);
		""")
		insert_arr = [(None,) * 8] * 2000  # 2000 - max number of games per season
		insert_row = [None] * 8
		res_changes = [0] * 8000
		row_index = 0

		for i in range(start_season, end_season - 1, -1):
			result = conn.execute("""
			SELECT COUNT(*)
			FROM games_momentum_new
				INNER JOIN games_table ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
			WHERE SEASON=?
			""", (i,))
			num_of_rows = result.fetchone()[0]
			games_stats = conn.execute("""SELECT games_table.GAME_TAG, games_momentum_new.POSSESSIONS_ARR, games_table.HOME_SCORE, games_table.AWAY_SCORE
			FROM games_momentum_new
				INNER JOIN games_table ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
			WHERE SEASON=?
			""", (i,))

			games_data = games_stats.fetchall()
			row_index = 0

			for k, game in enumerate(games_data):
				start_t = time.time()
				game_tag = game[0]
				game_possessions_1 = game[1]
				home_score = game[2]
				away_score = game[3]
				who_start = game[1][1]

				game_possessions_2 = [(int(_) if _[0] != "'" else '0') for _ in str.split(game_possessions_1, '|') if not (_[0] == "'" and _[1] != "0")]  # quarter and OT filtering
				# a_possessions = [_ for _ in game_possessions_2[0::2] if type(_) is int]
				# b_possessions = [_ for _ in game_possessions_2[1::2] if type(_) is int]
				a_possessions = game_possessions_2[0::2]
				b_possessions = game_possessions_2[1::2]

				a_poss_perm = GenPermReg(a_possessions)
				b_poss_perm = GenPermReg(b_possessions)

				if who_start == 'H':
					for j in range(8000):
						res_changes[j] = calc_possessions_change(a_poss_perm[j], b_poss_perm[j])
				else:
					for j in range(8000):
						res_changes[j] = calc_possessions_change(a_poss_perm[j], b_poss_perm[j])

				insert_row[0] = game_tag
				insert_row[1] = res_changes[0]
				insert_row[2] = np.median(res_changes)
				insert_row[3] = np.mean(res_changes)
				insert_row[4] = np.var(res_changes, ddof=1)
				insert_row[5] = sum([1 for _ in res_changes if _ < res_changes[0]]) / 8000
				insert_row[6] = sum([1 for _ in res_changes if _ < insert_row[2]])
				insert_row[7] = sum([1 for _ in res_changes if _ == insert_row[2]])
				insert_arr[row_index] = tuple(insert_row)

				row_index += 1

				if k % 50 == 0:
					print("k={} - {}".format(k, time.time() - start_t))

			conn.executemany("INSERT INTO games_changes_momentum3 VALUES (?,?,?,?,?,?,?,?)", insert_arr[0:row_index])
			conn.commit()
			print("finished season {}".format(i))


def calc_possessions_dist(start_season, end_season):
	with sqlite3.connect('Games.db') as conn:
		conn.execute("""
		CREATE TABLE IF NOT EXISTS games_poss_dist3(
			GAME_TAG TEXT NOT NULL UNIQUE,
			MAX_DIST REAL NOT NULL,
			MEDIAN REAL NOT NULL,
			MEAN REAL NOT NULL,
			VAR REAL NOT NULL,
			P_VALUE REAL NOT NULL,
			MEDIAN_SMALLER INT NOT NULL,
			MEDIAN_EQUAL INT NOT NULL
		);
		""")
		insert_arr = [(None,) * 8] * 2000  # 2000 - max number of games per season
		insert_row = [None] * 8
		res_changes = [0] * 8000
		row_index = 0

		for i in range(start_season, end_season - 1, -1):
			result = conn.execute("""
			SELECT COUNT(*)
			FROM games_momentum_new
				INNER JOIN games_table ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
				INNER JOIN games_scores on games_table.GAME_TAG = games_scores.GAME_TAG
			WHERE SEASON=? AND games_scores.VALID = 1
			""", (i,))
			num_of_rows = result.fetchone()[0]
			games_stats = conn.execute("""
			SELECT
				games_table.GAME_TAG, games_momentum_new.POSSESSIONS_ARR, games_table.HOME_SCORE, games_table.AWAY_SCORE
			FROM games_momentum_new
				INNER JOIN games_table ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
				INNER JOIN games_scores on games_table.GAME_TAG = games_scores.GAME_TAG
			WHERE SEASON=? AND games_scores.VALID = 1
			""", (i,))

			games_data = games_stats.fetchall()
			row_index = 0

			for k, game in enumerate(games_data):
				start_t = time.time()
				game_tag = game[0]
				game_possessions_1 = game[1]
				home_score = game[2]
				away_score = game[3]
				who_start = game[1][1]

				game_possessions_2 = [(int(_) if _[0] != "'" else '0') for _ in str.split(game_possessions_1, '|') if not (_[0] == "'" and _[1] != "0")]  # quarter and OT filtering
				# a_possessions = [_ for _ in game_possessions_2[0::2] if type(_) is int]
				# b_possessions = [_ for _ in game_possessions_2[1::2] if type(_) is int]
				a_possessions = game_possessions_2[0::2]
				b_possessions = game_possessions_2[1::2]

				a_poss_perm = GenPermReg(a_possessions)
				b_poss_perm = GenPermReg(b_possessions)

				if who_start == 'H':
					for j in range(8000):
						res_changes[j] = calc_max_point_distance(a_poss_perm[j], b_poss_perm[j], home_score, away_score)
				else:
					for j in range(8000):
						res_changes[j] = calc_max_point_distance(a_poss_perm[j], b_poss_perm[j], away_score, home_score)

				insert_row[0] = game_tag
				insert_row[1] = res_changes[0]
				insert_row[2] = np.median(res_changes)
				insert_row[3] = np.mean(res_changes)
				insert_row[4] = np.var(res_changes, ddof=1)
				insert_row[5] = sum([1 for _ in res_changes if _ <= res_changes[0]]) / 8000
				insert_row[6] = sum([1 for _ in res_changes if _ < insert_row[2]])
				insert_row[7] = sum([1 for _ in res_changes if _ == insert_row[2]])

				insert_arr[row_index] = tuple(insert_row)

				row_index += 1

				if k % 50 == 0:
					print("k={} - {}".format(k, time.time() - start_t))

			conn.executemany("INSERT INTO games_poss_dist3 VALUES (?,?,?,?,?,?,?,?)", insert_arr[0:row_index])
			conn.commit()
			print("finished season {}".format(i))


def check_bias():
	arr = "'A'|'1'|2|0|2|2|0|0|0|0|0|2|1|0|0|0|2|0|0|3|0|0|3|3|0|2|0|2|2|0|0|0|0|3|2|2|2|0|0|0|0|2|0|2|0|1|0|0|2|1|0|3|0|'2'|0|0|2|0|0|0|0|2|2|2|0|2|0|0|2|0|0|0|0|0|2|2|3|2|0|2|0|3|0|3|2|3|0|2|3|0|0|1|2|0|0|0|0|0|0|2|0|0|0|1|2|0|'3'|0|0|2|0|1|0|0|2|3|2|2|0|2|0|3|0|0|0|0|0|0|2|2|0|3|0|0|0|3|2|3|0|2|2|0|2|2|3|2|0|0|3|2|1|2|0|'4'|'0'|0|0|2|2|3|0|2|3|0|2|0|3|3|0|0|2|0|2|0|0|0|2|2|3|0|3|0|3|2|2|2|0|2|0|2|0|0|2|0|0|0|2|0|0|0|2|1|0|0|2|3"
	game_possessions_2 = [(int(_) if _[0] != "'" else '0') for _ in str.split(arr, '|') if not (_[0] == "'" and _[1] != "0")]  # quarter and OT filtering
	a_possessions = [_ for _ in game_possessions_2[0::2] if type(_) is int]
	b_possessions = [_ for _ in game_possessions_2[1::2] if type(_) is int]
	res_area = [0.] * 8000

	a_poss_perm = GenPermReg(a_possessions)
	b_poss_perm = GenPermReg(b_possessions)

	score_a = sum(a_possessions)
	score_b = sum(b_possessions)

	for i in range(8000):
		res_area[i] = calcArea(a_poss_perm[i], b_poss_perm[i], score_a, score_b)

	print("without 0:\t", res_area[0], np.mean(res_area), np.std(res_area))
	plt.hist(res_area, bins=100, rwidth=0.75)
	plt.show(block=False)

	arr = "'A'|'1'|2|0|2|2|0|0|0|0|0|2|1|0|0|0|2|0|0|3|0|0|3|3|0|2|0|2|2|0|0|0|0|3|2|2|2|0|0|0|0|2|0|2|0|1|0|0|2|1|0|3|0|'2'|0|0|2|0|0|0|0|2|2|2|0|2|0|0|2|0|0|0|0|0|2|2|3|2|0|2|0|3|0|3|2|3|0|2|3|0|0|1|2|0|0|0|0|0|0|2|0|0|0|1|2|0|'3'|0|0|2|0|1|0|0|2|3|2|2|0|2|0|3|0|0|0|0|0|0|2|2|0|3|0|0|0|3|2|3|0|2|2|0|2|2|3|2|0|0|3|2|1|2|0|'4'|'0'|0|0|2|2|3|0|2|3|0|2|0|3|3|0|0|2|0|2|0|0|0|2|2|3|0|3|0|3|2|2|2|0|2|0|2|0|0|2|0|0|0|2|0|0|0|2|1|0|0|2|3"
	game_possessions_2 = [(int(_) if _[0] != "'" else '0') for _ in str.split(arr, '|') if not (_[0] == "'" and _[1] != "0")]  # quarter and OT filtering
	a_possessions = [_ for _ in game_possessions_2[0::2] if type(_) is int] + [0] * 400
	b_possessions = [_ for _ in game_possessions_2[1::2] if type(_) is int] + [0] * 400
	res_area = [0.] * 8000

	a_poss_perm = GenPermReg(a_possessions)
	b_poss_perm = GenPermReg(b_possessions)

	score_a = sum(a_possessions)
	score_b = sum(b_possessions)

	for i in range(8000):
		res_area[i] = calcArea(a_poss_perm[i], b_poss_perm[i], score_a, score_b)

	print("with 0:\t", res_area[0], np.mean(res_area), np.std(res_area))
	plt.figure()
	plt.hist(res_area, bins=100, rwidth=0.75)
	plt.show()


def calc_area_corr():
	# arr = "'A'|'1'|0|2|2|0|0|3|0|0|2|0|2|0|3|0|0|0|0|3|0|0|2|2|0|3|0|2|3|3|0|2|0|0|0|2|2|2|3|3|3|2|3|0|2|2|2|2|2|0|0|0|'2'|'0'|0|0|2|0|3|3|2|0|2|0|2|0|2|0|0|2|3|3|1|0|0|2|0|1|2|0|0|2|0|1|2|0|0|0|0|2|2|2|0|2|2|0|2|3|2|2|2|2|0|0|0|0|0|0|'3'|0|0|2|2|0|0|0|0|0|0|2|0|0|2|3|0|0|3|2|0|2|0|0|0|3|2|0|0|2|3|2|0|3|3|1|0|2|0|0|0|2|3|0|0|0|2|2|0|2|3|3|0|0|0|'4'|'0'|2|2|0|2|0|2|0|2|2|2|2|3|2|2|3|0|3|2|0|2|2|0|0|0|2|2|2|0|2|0|3|0|2|0|0|0|2|0|0|2|2|0|2|0|3|0|0|2|0"
	arr = "'H'|'1'|0|0|2|3|2|2|0|2|0|0|2|0|0|0|1|0|2|0|3|3|0|0|3|2|0|0|0|3|2|0|3|2|0|0|0|0|0|0|2|0|0|2|0|2|0|3|0|0|0|0|2|2|2|0|'2'|'0'|2|2|2|0|3|2|0|3|2|2|3|0|3|0|0|0|0|0|0|0|2|1|0|0|1|0|2|0|2|0|1|2|2|3|0|1|0|1|2|0|0|2|0|1|0|0|'3'|0|1|0|2|0|3|0|3|0|2|0|0|0|2|2|3|0|0|3|2|0|2|0|2|0|0|3|2|0|2|2|0|2|3|2|0|0|2|2|3|1|3|0|2|3|0|0|'4'|2|0|0|0|0|3|0|3|2|0|1|0|3|3|3|2|2|0|2|0|0|0|2|0|0|2|0|2|2|0|0|2|0|0|0|0|0|0|2|3|0|0|3|0|0"
	game_possessions_2 = [(int(_) if _[0] != "'" else '0') for _ in str.split(arr, '|') if not (_[0] == "'" and _[1] != "0")]  # quarter and OT filtering
	a_possessions = [_ for _ in game_possessions_2[0::2] if type(_) is int]
	b_possessions = [_ for _ in game_possessions_2[1::2] if type(_) is int]
	res_area = [0.] * 8000
	res_run = [0] * 8000
	max_dist = [0.] * 8000

	a_poss_perm = GenPermReg(a_possessions)
	b_poss_perm = GenPermReg(b_possessions)

	score_a = sum(a_possessions)
	score_b = sum(b_possessions)

	for i in range(8000):
		res_area[i] = calcArea(a_poss_perm[i], b_poss_perm[i], score_a, score_b)
		res_run[i] = calc_possessions_change(a_poss_perm[i], b_poss_perm[i])
		max_dist[i] = calc_max_point_distance(a_poss_perm[i], b_poss_perm[i], score_a, score_b)

	plt.plot(res_run, res_area, 'o', markersize=2, zorder=0)
	# plt.plot(1/((np.array(res_run) + np.array(res_run)**0.5)**0.5), res_area, 'o', markersize=2, zorder=0)

	run_area_pol, run_area_pol_std_error, _, _, _ = np.polyfit(res_run, res_area, 4, full=True)
	run_area_pol_std_error = np.sqrt(run_area_pol_std_error[0] / (len(res_run)-1))
	run_area_pol_func = np.poly1d(run_area_pol)
	x_axis = np.linspace(min(res_run), max(res_run), 20)
	print(scipy.stats.pearsonr(run_area_pol_func(res_run), res_area), run_area_pol, run_area_pol_std_error)
	# print(scipy.stats.pearsonr(1/((np.array(res_run) + np.array(res_run)**0.5)**0.5), res_area), run_area_pol, run_area_pol_std_error)
	print(scipy.stats.spearmanr(res_run, res_area))
	#plt.errorbar(x_axis, run_area_pol_func(x_axis), yerr=run_area_pol_std_error, capsize=5, zorder=5)
	plt.show(block=False)

	plt.figure()
	plt.plot(max_dist, res_area, 'o', markersize=2)

	dist_area_pol, dist_area_pol_std_error, _, _, _ = np.polyfit(max_dist, res_area, 4, full=True)
	dist_area_pol_std_error = np.sqrt(dist_area_pol_std_error[0] / (len(max_dist)-1))
	dist_area_pol_func = np.poly1d(dist_area_pol)
	x_axis = np.linspace(min(max_dist), max(max_dist), 20)
	print(scipy.stats.pearsonr(dist_area_pol_func(max_dist), res_area), dist_area_pol, dist_area_pol_std_error)
	plt.errorbar(x_axis, dist_area_pol_func(x_axis), yerr=dist_area_pol_std_error, capsize=5, zorder=5)
	plt.show(block=False)

	plt.figure()
	plt.hist(res_run, bins=40, rwidth=0.75, density=True)
	plt.show()


def plot_graphs():
	with sqlite3.connect('Games.db') as conn:
		result = conn.execute("""
		SELECT COUNT(*)
		FROM games_area_momentum5
			INNER JOIN games_table ON games_area_momentum5.GAME_TAG = games_table.GAME_TAG
			INNER JOIN games_poss_dist3 ON games_poss_dist3.GAME_TAG = games_table.GAME_TAG
			INNER JOIN games_changes_momentum3 ON games_changes_momentum3.GAME_TAG = games_table.GAME_TAG
			INNER JOIN games_scores ON games_scores.GAME_TAG = games_table.GAME_TAG
		WHERE games_scores.VALID = 1
		""")
		num_of_rows = result.fetchone()[0]
		games_stats = conn.execute("""
		SELECT
			games_table.GAME_TAG, games_table.SEASON, games_table.YEAR, games_table.MONTH, games_table.HOME_SCORE, games_table.AWAY_SCORE,
			games_area_momentum5.AREA, games_area_momentum5.MEDIAN, games_area_momentum5.MEAN, games_area_momentum5.VAR, games_area_momentum5.MEDIAN_SMALLER, games_area_momentum5.MEDIAN_EQUAL,
			games_poss_dist3.MAX_DIST, games_poss_dist3.MEDIAN, games_poss_dist3.MEAN, games_poss_dist3.VAR, games_poss_dist3.MEDIAN_SMALLER, games_poss_dist3.MEDIAN_EQUAL,
			games_changes_momentum3.CHANGES_COUNT, games_changes_momentum3.MEDIAN, games_changes_momentum3.MEAN, games_changes_momentum3.VAR, games_changes_momentum3.MEDIAN_SMALLER, games_changes_momentum3.MEDIAN_EQUAL
		FROM games_area_momentum5
			INNER JOIN games_table ON games_area_momentum5.GAME_TAG = games_table.GAME_TAG
			INNER JOIN games_poss_dist3 ON games_poss_dist3.GAME_TAG = games_table.GAME_TAG
			INNER JOIN games_changes_momentum3 ON games_changes_momentum3.GAME_TAG = games_table.GAME_TAG
			INNER JOIN games_scores ON games_scores.GAME_TAG = games_table.GAME_TAG
		WHERE games_scores.VALID = 1
		""")

		games_data = games_stats.fetchall()
		stats_list = np.array([_[6:] for _ in games_data])

		#region Momentum values distribution

		fig, ax = plt.subplots(3, 1)

		result = conn.execute("""
		SELECT games_momentum_new.GAME_TAG, games_momentum_new.POSSESSIONS_ARR, games_table.HOME_SCORE, games_table.AWAY_SCORE
		FROM games_momentum_new
			INNER JOIN games_table ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
			INNER JOIN games_scores ON games_scores.GAME_TAG = games_table.GAME_TAG
		WHERE games_scores.VALID = 1
		""")

		games_data_2 = result.fetchall()
		row_index = 0
		res_value = [0.] * 8000

		for k, game in enumerate(games_data_2):
			#if game[0] != '201610260BOS':
			if game[0] != '201610260MEM':
				continue
			start_t = time.time()
			game_tag = game[0]
			game_possessions_1 = game[1]
			home_score = game[2]
			away_score = game[3]
			who_start = game[1][1]

			game_possessions_2 = [(int(_) if _[0] != "'" else '0') for _ in str.split(game_possessions_1, '|') if not (_[0] == "'" and _[1] != "0")]  # quarter and OT filtering
			# a_possessions = [_ for _ in game_possessions_2[0::2] if type(_) is int]
			# b_possessions = [_ for _ in game_possessions_2[1::2] if type(_) is int]
			a_possessions = game_possessions_2[0::2]
			b_possessions = game_possessions_2[1::2]

			a_poss_perm = GenPermReg(a_possessions)
			b_poss_perm = GenPermReg(b_possessions)

			if who_start == 'H':
				for j in range(8000):
					res_value[j] = calcArea(a_poss_perm[j], b_poss_perm[j], home_score, away_score)
			else:
				for j in range(8000):
					res_value[j] = calcArea(a_poss_perm[j], b_poss_perm[j], away_score, home_score)

			ax[0].hist(res_value, bins=100, rwidth=0.75, density=True, color='C0')
			ax[0].set_title(r"$H_0$ area distribution")
			ax[0].vlines(np.median(res_value), 0, 0.0035, color='r', linestyles='--', label='Median')
			ax[0].vlines(np.mean(res_value), 0, 0.0035, color='y', linestyles='--', label='Mean')
			ax[0].vlines(res_value[0], 0, 0.0035, color='k', linestyles='--', label='Original')
			ax[0].legend()

			if who_start == 'H':
				for j in range(8000):
					res_value[j] = calc_max_point_distance(a_poss_perm[j], b_poss_perm[j], home_score, away_score)
			else:
				for j in range(8000):
					res_value[j] = calc_max_point_distance(a_poss_perm[j], b_poss_perm[j], away_score, home_score)

			ax[1].hist(res_value, bins=100, rwidth=0.75, density=True, color='C1')
			ax[1].set_title(r"$H_0$ max distance distribution")
			ax[1].vlines(np.median(res_value), 0, 0.2, color='r', linestyles='--', label='Median')
			ax[1].vlines(np.mean(res_value), 0, 0.2, color='blue', linestyles='--', label='Mean')
			ax[1].vlines(res_value[0], 0, 0.2, color='k', linestyles='--', label='Original')
			ax[1].legend()

			if who_start == 'H':
				for j in range(8000):
					res_value[j] = calc_possessions_change(a_poss_perm[j], b_poss_perm[j])
			else:
				for j in range(8000):
					res_value[j] = calc_possessions_change(a_poss_perm[j], b_poss_perm[j])

			ax[2].hist(res_value, bins=100, rwidth=0.75, density=True, color='C2')
			ax[2].set_title(r"$H_0$ runs count distribution")
			ax[2].vlines(np.median(res_value), 0, 0.35, color='r', linestyles='--', label='Median')
			ax[2].vlines(np.mean(res_value), 0, 0.35, color='blue', linestyles='--', label='Mean')
			ax[2].vlines(res_value[0], 0, 0.35, color='k', linestyles='--', label='Original')
			ax[2].legend()

			# insert_row[0] = game_tag
			# insert_row[1] = res_area[0]
			# insert_row[2] = np.median(res_area)
			# insert_row[3] = np.mean(res_area)
			# insert_row[4] = np.var(res_area, ddof=1)
			# insert_row[5] = sum([1 for _ in res_area if _ < res_area[0]]) / 8000
		#endregion

		fig, ax = plt.subplots(3, 1)

		#region Area
		# Binom
		area_greater_than_median = sum(stats_list[:, 0] >= stats_list[:, 1])
		area_greater_than_median_percentage = area_greater_than_median / len(stats_list[:, 0])
		expected_mean = np.sum(8000 - stats_list[:, 4]) / 8000
		expected_std = np.sqrt(np.sum((8000 - stats_list[:, 4])*stats_list[:, 4]) / (8000*8000))
		print(area_greater_than_median, stats_list.shape[0])
		print((area_greater_than_median - expected_mean)/expected_std)
		print(scipy.stats.norm.sf(area_greater_than_median, loc=expected_mean, scale=expected_std))
		# std_count = (area_greater_than_median - scipy.stats.binom.mean(len(stats_list[:, 0]), 0.498)) / scipy.stats.binom.std(len(stats_list[:, 0]), 0.498)
		# print(len(stats_list[:, 0]), area_greater_than_median, std_count, scipy.stats.binom_test(area_greater_than_median, len(stats_list[:, 0]), 0.498, alternative='greater'))

		# Z-score
		area_z_score = (stats_list[:, 0] - stats_list[:, 2]) / np.sqrt(stats_list[:, 3])
		area_z_score_mean = np.mean(area_z_score)
		area_z_score_std_of_sample_mean = np.std(area_z_score, ddof=1)/np.sqrt(len(area_z_score))
		std_count = area_z_score_mean / area_z_score_std_of_sample_mean
		print(area_z_score_mean, std_count)
		ax[0].hist(area_z_score, bins=100, rwidth=0.75, density=True, color='C0')
		ax[0].set_title("Area Z-Score")
		ax[0].vlines(area_z_score_mean, 0, 0.6, color='k', linestyles='--', label='Mean')
		ax[0].annotate(r"${}\sigma_{{\bar{{X}}}}$".format(round(std_count, 1)), xy=(area_z_score_mean, 0.45), xytext=(area_z_score_mean+1, 0.42), arrowprops=dict(facecolor='black', shrink=0.1, width=0.2, headwidth=4), fontsize=15)
		ax[0].legend()
		#endregion

		#region Max dist
		# Binom
		max_dist_greater_than_median = sum(stats_list[:, 6] >= stats_list[:, 7])
		expected_mean = np.sum(8000 - stats_list[:, 10]) / 8000
		expected_std = np.sqrt(np.sum((8000 - stats_list[:, 10])*stats_list[:, 10]) / (8000*8000))
		print()
		print(max_dist_greater_than_median, stats_list.shape[0])
		print((max_dist_greater_than_median - expected_mean)/expected_std)
		print(scipy.stats.norm.sf(max_dist_greater_than_median, loc=expected_mean, scale=expected_std))
		# max_dist_greater_than_median_percentage = max_dist_greater_than_median / len(stats_list[:, 4])
		# std_count = (max_dist_greater_than_median - scipy.stats.binom.mean(len(stats_list[:, 4]), 0.501)) / scipy.stats.binom.std(len(stats_list[:, 4]), 0.501)
		# print(len(stats_list[:, 4]), max_dist_greater_than_median, std_count, scipy.stats.binom_test(max_dist_greater_than_median, len(stats_list[:, 4]), 0.501, alternative='two-sided'))

		# Z-score
		max_dist_z_score = (stats_list[:, 6] - stats_list[:, 8]) / np.sqrt(stats_list[:, 9])
		max_dist_z_score_mean = np.mean(max_dist_z_score)
		max_dist_z_score_std_of_sample_mean = np.std(max_dist_z_score, ddof=1) / np.sqrt(len(max_dist_z_score))
		std_count = max_dist_z_score_mean / max_dist_z_score_std_of_sample_mean
		print(max_dist_z_score_mean, std_count)
		ax[1].hist(max_dist_z_score, bins=100, rwidth=0.75, density=True, color='C1')
		ax[1].set_title("Max distance Z-Score")
		ax[1].vlines(max_dist_z_score_mean, 0, 0.6, color='k', linestyles='--', label='Mean')
		ax[1].annotate(r"${}\sigma_{{\bar{{X}}}}$".format(round(std_count, 1)), xy=(max_dist_z_score_mean, 0.45), xytext=(max_dist_z_score_mean+1, 0.42), arrowprops=dict(facecolor='black', shrink=0.1, width=0.2, headwidth=4), fontsize=15)
		ax[1].legend()
		#endregion

		#region No. runs
		# Binom
		runs_count_smaller_than_median = sum(stats_list[:, 12] < stats_list[:, 13])
		expected_mean = np.sum(stats_list[:, 16]) / 8000
		expected_std = np.sqrt(np.sum((8000 - stats_list[:, 16])*stats_list[:, 16]) / (8000*8000))
		print()
		print(runs_count_smaller_than_median, stats_list.shape[0])
		print((runs_count_smaller_than_median - expected_mean)/expected_std)
		print(scipy.stats.norm.sf(runs_count_smaller_than_median, loc=expected_mean, scale=expected_std))

		# runs_count_greater_than_median_percentage = runs_count_greater_than_median / len(stats_list[:, 8])
		# std_count = (runs_count_greater_than_median - scipy.stats.binom.mean(len(stats_list[:, 8]), 0.45)) / scipy.stats.binom.std(len(stats_list[:, 8]), 0.45)
		# print(len(stats_list[:, 8]), runs_count_greater_than_median, std_count, scipy.stats.binom_test(runs_count_greater_than_median, len(stats_list[:, 8]), 0.45, alternative='greater'))

		# Z-score
		runs_count_z_score = (stats_list[:, 12] - stats_list[:, 14]) / np.sqrt(stats_list[:, 15])
		runs_count_z_score_mean = np.mean(runs_count_z_score)
		runs_count_z_score_std_of_sample_mean = np.std(runs_count_z_score, ddof=1) / np.sqrt(len(runs_count_z_score))
		std_count = runs_count_z_score_mean / runs_count_z_score_std_of_sample_mean
		print(runs_count_z_score_mean, std_count)
		ax[2].hist(runs_count_z_score, bins=100, rwidth=0.75, density=True, color='C2')
		ax[2].set_title("Runs count Z-Score")
		ax[2].vlines(runs_count_z_score_mean, 0, 0.6, color='k', linestyles='--', label='Mean')
		ax[2].annotate(r"${}\sigma_{{\bar{{X}}}}$".format(round(std_count, 1)), xy=(runs_count_z_score_mean, 0.45), xytext=(runs_count_z_score_mean+1, 0.42), arrowprops=dict(facecolor='black', shrink=0.1, width=0.2, headwidth=4), fontsize=15)
		ax[2].legend()
		#endregion

		plt.show(block=False)

		#region Binom per season

		area_binom = [0.] * 17
		area_expected_mean = [0.] * 17
		area_expected_std = [0.] * 17
		max_dist_binom = [0.] * 17
		max_dist_expected_mean = [0.] * 17
		max_dist_expected_std = [0.] * 17
		runs_binom = [0.] * 17
		runs_expected_mean = [0.] * 17
		runs_expected_std = [0.] * 17

		for i in range(2002, 2019):
			cur_season_stats = np.array([_[6:] for _ in games_data if _[1] == i])
			area_binom[i-2002] = sum(cur_season_stats[:, 0] >= cur_season_stats[:, 1]) / len(cur_season_stats)
			area_expected_mean[i-2002] = sum(1 - (cur_season_stats[:, 4]/8000)) / len(cur_season_stats)
			max_dist_binom[i-2002] = sum(cur_season_stats[:, 6] >= cur_season_stats[:, 7]) / len(cur_season_stats)
			max_dist_expected_mean[i - 2002] = sum(1 - (cur_season_stats[:, 10]/8000)) / len(cur_season_stats)
			runs_binom[i-2002] = sum(cur_season_stats[:, 12] < cur_season_stats[:, 13]) / len(cur_season_stats)
			runs_expected_mean[i - 2002] = sum(cur_season_stats[:, 16] / 8000) / len(cur_season_stats)

		plt.figure()
		plt.plot(range(2002, 2019), area_binom, '-o', label='Area')
		plt.plot(range(2002, 2019), max_dist_binom, '-o', label='Max dist')
		plt.plot(range(2002, 2019), runs_binom, '-o', label='Runs')

		# plt.plot([2002, 2018], [0.498, 0.498], 'b--', label='Expected for Area')
		# plt.plot([2002, 2018], [0.501, 0.501], 'r--', label='Expected for Max dist')
		# plt.plot([2002, 2018], [0.45, 0.45], 'g--', label='Expected for Runs')
		plt.plot(range(2002, 2019), area_expected_mean, 'b--', label='Expected for Area')
		plt.plot(range(2002, 2019), max_dist_expected_mean, 'r--', label='Expected for Max dist')
		plt.plot(range(2002, 2019), runs_expected_mean, 'g--', label='Expected for Runs')

		plt.legend(ncol=2, loc=1)
		plt.ylim(ymax=0.7)
		plt.title("Binomial test success proportion per season")
		plt.xlabel('Season')
		plt.ylabel('Success proportion')
		plt.subplots_adjust(right=0.7)
		# plt.show(block=False)
		#endregion
		plt.show()
		return

		# region Features mean prob

		result = conn.execute("""
		SELECT MEDIAN_LESS, MEDIAN_EQUAL
		FROM games_changes_momentum2
		INNER JOIN games_scores ON games_scores.GAME_TAG = games_changes_momentum2.GAME_TAG
		WHERE VALID = 1
		""")

		result_data = result.fetchall()
		median_greater = np.array([_[0] for _ in result_data])
		median_equal = np.array([_[1] for _ in result_data])
		prob_arr = 1 - (median_greater + median_equal)/8000
		print("median prob mean: {},\tstd: {}".format(np.median(prob_arr), np.std(prob_arr)))
		plt.figure()
		plt.hist(prob_arr, bins=20, rwidth=0.75, density=True)
		plt.show(block=False)

		# endregion

		#region OT prob

		result = conn.execute("""
		SELECT COUNT(*)
		FROM games_ot_prob
			INNER JOIN games_table ON games_ot_prob.GAME_TAG = games_table.GAME_TAG
		""")
		num_of_rows = result.fetchone()[0]
		games_stats = conn.execute("""
		SELECT
			games_table.GAME_TAG, games_table.SEASON, games_table.YEAR, games_table.MONTH, games_table.HOME_SCORE, games_table.AWAY_SCORE,
			games_ot_prob.WHO_TIED, games_ot_prob.WHO_WON
		FROM games_ot_prob
			INNER JOIN games_table ON games_ot_prob.GAME_TAG = games_table.GAME_TAG
		""")

		games_data_ot = games_stats.fetchall()
		# area_hist(games_data)
		stats_list_ot = np.array([_[6:] for _ in games_data_ot])

		unconditional_prob = sum(stats_list_ot[:, 0] == stats_list_ot[:, 1]) / num_of_rows
		prob_std = np.sqrt(unconditional_prob*(1-unconditional_prob)/num_of_rows)

		# H_tied = np.array([_ for _ in stats_list if _[0] == 'H'])
		# A_tied = np.array([_ for _ in stats_list if _[0] == 'A'])
		H_won = stats_list_ot[stats_list_ot[:, 1] == 'H']
		A_won = stats_list_ot[stats_list_ot[:, 1] == 'A']
		H_tied = stats_list_ot[stats_list_ot[:, 0] == 'H']
		A_tied = stats_list_ot[stats_list_ot[:, 0] == 'A']

		# num_of_H_tied = len(H_tied)
		num_of_H_won = H_won.shape[0]
		num_of_A_won = num_of_rows - num_of_H_won
		num_of_H_tied = H_tied.shape[0]
		num_of_A_tied = num_of_rows - num_of_H_tied

		# print(num_of_H_won)
		H_won_prob = num_of_H_won / num_of_rows
		A_won_prob = 1 - H_won_prob
		unconditional_H_prob = sum(H_tied[:, 0] == H_tied[:, 1]) / num_of_H_tied
		unconditional_A_prob = sum(A_tied[:, 0] == A_tied[:, 1]) / num_of_A_tied
		prob_H_won_std = np.sqrt(H_won_prob * (1-H_won_prob)/num_of_rows)
		prob_H_tied_std = np.sqrt(unconditional_H_prob*(1-unconditional_H_prob)/num_of_H_tied)
		prob_A_tied_std = np.sqrt(unconditional_A_prob * (1 - unconditional_A_prob) / num_of_A_tied)

		# tied_won_arr = [H_won_prob, unconditional_prob, unconditional_H_prob, unconditional_A_prob]
		home_tied_arr = [H_won_prob, 1 - unconditional_A_prob, unconditional_H_prob]
		away_tied_arr = [A_won_prob, 1 - unconditional_H_prob, unconditional_A_prob]
		# tied_lose_arr = [1 - _ for _ in tied_won_arr]
		probs_std = [2*prob_H_won_std, [2*prob_H_tied_std, 2*prob_A_tied_std]]

		ind = np.arange(3)  # the x locations for the groups
		width = 0.2  # the width of the bars

		plt.figure()
		plt.bar((ind[1:] - width), [home_tied_arr[1], away_tied_arr[1]], width, yerr=probs_std[1], capsize=4, color='green', label='Won after other team tied')
		plt.bar(ind[1:], [home_tied_arr[0], away_tied_arr[0]], width, yerr=probs_std[0], capsize=4, color='blue', label='Won regardless of who tied')
		plt.bar((ind[1:] + width), [home_tied_arr[2], away_tied_arr[2]], width, yerr=probs_std[1], capsize=4, color='purple', label='Won after tied')
		plt.bar(ind[0] - width / 2, 1 - unconditional_prob, width, yerr=prob_std, capsize=4, color='green')
		plt.bar(ind[0] + width / 2, unconditional_prob, width, yerr=prob_std, capsize=4, color='purple')
		plt.xticks(ind, ['Any team\n(Unconditional)', 'Home', 'Away'])
		plt.ylim(ymax=1)
		plt.legend(loc=2)
		plt.show(block=False)
		#endregion

		#region Series Prob

		result = conn.execute("""
		SELECT COUNT(*)
		FROM series_table
		""")
		num_of_rows = result.fetchone()[0]
		result = conn.execute("""
		SELECT COUNT(*)
		FROM series_table
		GROUP BY SERIES_ID
		ORDER BY SERIES_ID ASC
		""")
		num_per_series = [_[0] for _ in result.fetchall()]
		num_of_series = len(num_per_series)
		games_stats = conn.execute("""
		SELECT *
		FROM series_table
		ORDER BY SERIES_ID, GAME_NUM ASC
		""")
		games_series_data = games_stats.fetchall()

		# 0: 1-1, 1: 2-2, 2: 3-3
		# After the away team tied, what is the prob they'll win as home:
		# The condition is The away(before tie) team tied, and now they change fields (away becomes home)
		away_won_after_tie_lose_arr = np.zeros(3)
		away_tie_lose_count_arr = np.zeros(3)
		away_won_after_tie_win_arr = np.zeros(3)
		away_tie_win_count_arr = np.zeros(3)
		home_won_after_tie_arr = np.zeros(3)
		tie_count_arr = np.zeros(3)

		cur_index = 0
		cur_series = 1
		for series_games_count in num_per_series:
			cur_score = [0] * 2
			last_won = -1
			cur_won = -1
			series_dict = {games_series_data[cur_index][5]: 0, games_series_data[cur_index][6]: 1}
			for i in range(series_games_count):
				if games_series_data[cur_index + i][7] >= games_series_data[cur_index + i][8]:
					cur_won = series_dict[games_series_data[cur_index + i][5]]
				else:
					cur_won = series_dict[games_series_data[cur_index + i][6]]
				cur_score[cur_won] += 1
				if cur_score[cur_won] - 1 == cur_score[1-cur_won] and 2 <= cur_score[cur_won] <= 4:  # after tie
					tie_count_arr[cur_score[cur_won] - 2] += 1
					if cur_won == series_dict[games_series_data[cur_index + i][5]]:
						home_won_after_tie_arr[cur_score[cur_won] - 2] += 1

					if series_dict[games_series_data[cur_index + i - 1][5]] == series_dict[games_series_data[cur_index + i][6]]:
						if last_won == series_dict[games_series_data[cur_index + i - 1][6]]:
							away_tie_lose_count_arr[cur_score[cur_won] - 2] += 1
							if cur_won == 1 - last_won:
								away_won_after_tie_lose_arr[cur_score[cur_won] - 2] += 1
						else:
							away_tie_win_count_arr[cur_score[cur_won] - 2] += 1
							if cur_won == last_won:
								away_won_after_tie_win_arr[cur_score[cur_won] - 2] += 1

				last_won = cur_won
			cur_index += series_games_count
			cur_series += 1

		away_won_after_tie_lose_prob_arr = away_won_after_tie_lose_arr / away_tie_lose_count_arr
		away_won_after_tie_win_prob_arr = away_won_after_tie_win_arr / away_tie_win_count_arr
		home_won_after_tie_prob_arr = home_won_after_tie_arr / tie_count_arr
		away_won_after_tie_lose_std_arr = np.sqrt(away_won_after_tie_lose_prob_arr*(1-away_won_after_tie_lose_prob_arr)/away_tie_lose_count_arr)
		away_won_after_tie_win_std_arr = np.sqrt(away_won_after_tie_win_prob_arr*(1-away_won_after_tie_win_prob_arr)/away_tie_win_count_arr)
		home_won_after_tie_std_arr = np.sqrt(home_won_after_tie_prob_arr*(1-home_won_after_tie_prob_arr)/tie_count_arr)

		swap_tmp = 1 - away_won_after_tie_lose_prob_arr[1:]
		away_won_after_tie_lose_prob_arr[1:] = 1 - away_won_after_tie_win_prob_arr[1:]  # prob of home to win after lose
		away_won_after_tie_win_prob_arr[1:] = swap_tmp  # prob of home to lose after win
		swap_tmp = away_won_after_tie_lose_std_arr[1:].copy()
		away_won_after_tie_lose_std_arr[1:] = away_won_after_tie_win_std_arr[1:]
		away_won_after_tie_win_std_arr[1:] = swap_tmp
		home_won_after_tie_prob_arr[0] = 1 - home_won_after_tie_prob_arr[0]


		ind = np.arange(len(away_won_after_tie_lose_prob_arr))  # the x locations for the groups
		width = 0.2  # the width of the bars

		plt.figure()
		plt.bar((ind - width), away_won_after_tie_lose_prob_arr, width, yerr=2*away_won_after_tie_lose_std_arr, capsize=4, color='green', label='Won after lose')
		plt.bar(ind, home_won_after_tie_prob_arr, width, yerr=2*home_won_after_tie_std_arr, capsize=4, color='blue', label='Won regardless of previous game')
		plt.bar((ind + width), away_won_after_tie_win_prob_arr, width, yerr=2*away_won_after_tie_win_std_arr, capsize=4, color='purple', label='Won after win')
		plt.xticks(ind, ('Away\n1-1', 'Home\n2-2', 'Home\n3-3'))
		plt.ylim(ymax=1)
		plt.legend(loc=2)
		plt.show(block=False)
		#endregion

		#region Power analysis - ROC

		result = conn.execute("""
		SELECT
		*
		FROM power_analysis
		""")

		result_data = result.fetchall()

		test_method_arr = ['Area', 'Max_Dist', 'Runs']
		test_method_color_arr = ['C0', 'C1', 'C2']
		delta_arr = [0.01, 0.03, 0.04, 0.06]
		alpha_samples_count = 100

		fig, ax_matrix = plt.subplots(2, 2)
		ax = ax_matrix.flat

		for i, delta in enumerate(delta_arr):
			for k, test_method in enumerate(test_method_arr):
				zero_art_games_count_list = np.array([_[3] for _ in result_data if _[0] == test_method and _[1] == 0])
				zero_bigger_than_median_list = np.array([_[4] for _ in result_data if _[0] == test_method and _[1] == 0])
				delta_art_games_count_list = np.array([_[3] for _ in result_data if _[0] == test_method and _[1] == delta])
				delta_bigger_than_median_list = np.array([_[4] for _ in result_data if _[0] == test_method and _[1] == delta])

				if len(zero_art_games_count_list) == 0 or len(delta_art_games_count_list) == 0:
					continue
				# if len(delta_art_games_count_list) == 0:
				# 	continue

				if test_method == 'Runs':
					zero_p_value = np.array([scipy.stats.binom_test(zero_art_games_count_list[k] - zero_bigger_than_median_list[k], zero_art_games_count_list[k], 0.45, alternative="greater") for k in range(len(zero_art_games_count_list))])
					delta_p_value = np.array([scipy.stats.binom_test(delta_art_games_count_list[k] - delta_bigger_than_median_list[k], delta_art_games_count_list[k], 0.45, alternative="greater") for k in range(len(delta_art_games_count_list))])
				elif test_method == 'Area':
					zero_p_value = np.array([scipy.stats.binom_test(zero_bigger_than_median_list[k], zero_art_games_count_list[k], 0.498, alternative="greater") for k in range(len(zero_art_games_count_list))])
					delta_p_value = np.array([scipy.stats.binom_test(delta_bigger_than_median_list[k], delta_art_games_count_list[k], 0.498, alternative="greater") for k in range(len(delta_art_games_count_list))])
				else:
					zero_p_value = np.array([scipy.stats.binom_test(zero_bigger_than_median_list[k], zero_art_games_count_list[k], 0.501, alternative="greater") for k in range(len(zero_art_games_count_list))])
					delta_p_value = np.array([scipy.stats.binom_test(delta_bigger_than_median_list[k], delta_art_games_count_list[k], 0.501, alternative="greater") for k in range(len(delta_art_games_count_list))])

				sensitivity = np.array([0.] * alpha_samples_count)
				fall_out = np.array([0.] * alpha_samples_count)

				for j, alpha in enumerate(np.linspace(0, 1, alpha_samples_count)):
					zero_positive_count = sum(zero_p_value <= alpha)
					delta_positive_count = sum(delta_p_value <= alpha)

					positive_count = len(delta_p_value)
					negative_count = len(zero_p_value)

					sensitivity[j] = delta_positive_count / positive_count
					# fall_out[j] = alpha
					fall_out[j] = zero_positive_count / negative_count

				_, unique_ind = np.unique(fall_out, return_index=True)
				fall_out_new = fall_out[unique_ind]
				sensitivity_new = sensitivity[unique_ind]

				ax[i].plot(fall_out_new, sensitivity_new, label=test_method.replace('_', ' '), linewidth=1, color=test_method_color_arr[k])

			ax[i].set_xlabel('1 - specificity')
			ax[i].set_ylabel('sensitivity')
			ax[i].set_title("ROC curve ($\Delta$ = {})".format(delta))
			ax[i].legend(loc=4)

		plt.tight_layout()
		plt.show(block=False)
		plt.savefig('ROC_img.tiff', format='tiff', dpi=300)
		plt.figure()

		for test_method in test_method_arr:
			zero_art_games_count_list = np.array([_[3] for _ in result_data if _[0] == test_method and _[1] == 0])
			zero_bigger_than_median_list = np.array([_[4] for _ in result_data if _[0] == test_method and _[1] == 0])

			if len(zero_art_games_count_list) == 0:
				continue

			if test_method == 'Area':
				zero_p_value = np.array([scipy.stats.binom_test(zero_bigger_than_median_list[k], zero_art_games_count_list[k], 0.498, alternative="greater") for k in range(len(zero_art_games_count_list))])
			elif test_method == 'Max_Dist':
				zero_p_value = np.array([scipy.stats.binom_test(zero_bigger_than_median_list[k], zero_art_games_count_list[k], 0.501, alternative="greater") for k in range(len(zero_art_games_count_list))])
			elif test_method == 'Runs':
				zero_p_value = np.array([scipy.stats.binom_test(zero_art_games_count_list[k] - zero_bigger_than_median_list[k], zero_art_games_count_list[k], 0.4512, alternative="greater") for k in range(len(zero_art_games_count_list))])

			sensitivity = np.array([0.] * alpha_samples_count)
			fall_out = np.array([0.] * alpha_samples_count)

			for j, alpha in enumerate(np.linspace(0, 1, alpha_samples_count)):
				zero_positive_count = sum(zero_p_value <= alpha)

				negative_count = len(zero_p_value)

				sensitivity[j] = zero_positive_count / negative_count
				fall_out[j] = alpha

			_, unique_ind = np.unique(fall_out, return_index=True)
			fall_out_new = fall_out[unique_ind]
			sensitivity_new = sensitivity[unique_ind]

			plt.plot(fall_out, sensitivity, label=test_method)
			# plt.figure()
			# plt.hist(zero_art_games_count_list - zero_bigger_than_median_list, label=test_method, rwidth=0.75)
			# plt.legend()

		plt.plot([0, 1], [0, 1], 'k--', label='Ideal')
		plt.xlabel("$\\alpha$")
		plt.ylabel('FPR')
		plt.title("False positives rate ($\Delta$ = 0)")
		plt.legend(loc=4)

		#endregion

		plt.show()


def calc_who_tied(poss_arr, who_start):
	h_a_map = {'H': 0, 'A': 1}
	h_a_inv_map = {0: 'H', 1: 'A'}
	j = h_a_map[who_start]

	if len(poss_arr) % 2 == 0:
		j = 1 - j

	for poss in poss_arr[::-1]:
		if type(poss) is not int:
			j = 1 - j
			continue

		if poss > 0:
			return h_a_inv_map[j]

		j = 1 - j

	return who_start


def calc_who_leads_most(poss_arr, who_start):
	h_a_map = {'H': 0, 'A': 1}
	h_a_inv_map = {0: 'H', 1: 'A'}

	a_b_count = [0, 0]
	cur_score = [0, 0]
	j = 0

	for poss in poss_arr:
		if type(poss) is not int:
			j = 1 - j
			continue

		if cur_score[0] != cur_score[1]:
			a_b_count[1 if cur_score[1] > cur_score[0] else 0] += 1
		
		cur_score[j] += poss
		j = 1 - j
	
	if a_b_count[0] != a_b_count[1]:
		return h_a_inv_map[h_a_map[who_start] if a_b_count[0] < a_b_count[1] else 1 - h_a_map[who_start]]
	else:
		return calc_who_tied(poss_arr, who_start)


def calc_ot_winning_prob():
	with sqlite3.connect('Games.db') as conn:
		# conn.execute("DROP TABLE IF EXISTS games_ot_prob")
		conn.execute("""
		CREATE TABLE IF NOT EXISTS games_ot_prob(
			GAME_TAG TEXT NOT NULL UNIQUE,
			WHO_TIED TEXT NOT NULL,
			WHO_WON TEXT NOT NULL
		);
		""")
		insert_arr = [(None,) * 3] * 2000  # 2000 - max number of games per season
		insert_row = [None] * 3
		res_area = [0.] * 8000
		row_index = 0

		h_a_map = {'H': 0, 'A': 1}
		h_a_inv_map = {0: 'H', 1: 'A'}

		for i in range(2017, 2001, -1):
			result = conn.execute("""
			SELECT COUNT(*)
			FROM games_momentum_new
				INNER JOIN games_table ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
			WHERE SEASON=? AND OT=1
			""", (i,))
			num_of_rows = result.fetchone()[0]
			games_stats = conn.execute("""SELECT games_table.GAME_TAG, games_momentum_new.POSSESSIONS_ARR, games_table.HOME_SCORE, games_table.AWAY_SCORE
			FROM games_momentum_new
				INNER JOIN games_table ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
			WHERE SEASON=? AND OT=1
			""", (i,))

			games_data = games_stats.fetchall()
			row_index = 0

			for k, game in enumerate(games_data):
				start_t = time.time()
				game_tag = game[0]
				game_possessions_1 = game[1]
				home_score = game[2]
				away_score = game[3]
				who_start = game[1][1]

				poss_arr = str.split(game_possessions_1, '|')
				q4_start = poss_arr.index("'4'")
				ot1_start = poss_arr.index("'1OT'")
				q4_possessions = [(int(_) if _[0] != "'" else '0') for _ in poss_arr[(q4_start+1):ot1_start] if not (_[0] == "'" and _[1] != "0")]  # quarter and OT filtering
				q4_start -= 4  # index after filtering
				ot1_start -= 5  # index after filtering
				who_start_q4 = who_start
				if q4_start % 2 == 1:
					who_start_q4 = h_a_inv_map[1 - h_a_map[who_start]]

				insert_row[0] = game_tag
				insert_row[1] = calc_who_leads_most(q4_possessions, who_start_q4)
				insert_row[2] = 'H' if home_score > away_score else 'A'

				insert_arr[row_index] = tuple(insert_row)

				row_index += 1

				if k % 50 == 0:
					print("k={} - {}".format(k, time.time() - start_t))

			conn.executemany("INSERT INTO games_ot_prob VALUES (?,?,?)", insert_arr[0:row_index])
			conn.commit()
			print("finished season {}".format(i))


def check_possessions():
	with sqlite3.connect('Games.db') as conn:
		games_stats = conn.execute("""SELECT games_table.GAME_TAG, games_momentum_new.POSSESSIONS_ARR, games_table.OT, games_scores.*
		FROM games_table
			INNER JOIN games_momentum_new ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
			INNER JOIN games_scores ON games_table.GAME_TAG == games_scores.GAME_TAG
		WHERE games_table.SEASON = 2018
		""")

		games_data = games_stats.fetchall()
		games_count = len(games_data)
		insert_arr = [(None,)*4] * games_count
		insert_row = [None] * 4

		for k, game in enumerate(games_data):
			game_tag = game[0]
			possessions = [(-1 if _[0] == "'" and _ != "'0'" else (int(_) if _[0] != "'" else '0')) for _ in game[1][8:].split('|')]
			ot_count = game[2] if game[2] is not None else 0

			who_start = game[1][1]

			j = 0 if who_start == 'H' else 1
			cur_part = 0
			insert_row = [None] * 4
			h_total_score, a_total_score = 0, 0
			h_partial_score, a_partial_score = 0, 0

			for play in possessions:
				if play == '0':
					j = 1 - j
					continue
				elif play == -1:
					true_score = list(map(int, game[6 + cur_part].split('-')))
					if h_partial_score != true_score[0] or a_partial_score != true_score[1]:
						insert_row[0] = 0
						insert_row[1] = cur_part
						insert_row[2] = "{}-{}".format(h_partial_score, a_partial_score)
						break
					cur_part += 1
					h_partial_score = 0
					a_partial_score = 0
				else:
					if j == 0:
						h_partial_score += play
						h_total_score += play
					else:
						a_partial_score += play
						a_total_score += play
					j = 1 - j

			if insert_row[0] is None:
				true_score = list(map(int, game[6 + cur_part].split('-')))
				if h_partial_score != true_score[0] or a_partial_score != true_score[1] or h_total_score != game[4] or a_total_score != game[5]:
					insert_row[0] = 0
					insert_row[1] = cur_part
					insert_row[2] = "{}-{}".format(h_partial_score, a_partial_score)
					if game_tag == '201503160GSW':
						print( game[1])
				else:
					insert_row[0] = 1
					insert_row[1] = None
					insert_row[2] = None

			insert_row[3] = game_tag

			insert_arr[k] = tuple(insert_row)

		conn.executemany("""UPDATE games_scores SET VALID=?, PROBLEMATIC_PART=?, PROBLEMATIC_SCORE=? WHERE GAME_TAG = ?""", insert_arr)


def gen_artificial_games(selected_games, delta):
	# with sqlite3.connect('Games.db') as conn:
	#
	# 	# conn.execute("DROP TABLE IF EXISTS artificial_games")
	# 	conn.execute("""
	# 	CREATE TABLE IF NOT EXISTS artificial_games(
	# 		GAME_TAG TEXT NOT NULL,
	# 		POSS_ARR TEXT NOT NULL,
	# 		DELTA REAL NOT NULL,
	# 		H_SCORE INT NOT NULL,
	# 		A_SCORE INT NOT NULL,
	# 		H_SUCCESS_PROB REAL NOT NULL,
	# 		A_SUCCESS_PROB REAL NOT NULL
	# 	);
	# 	""")
	# 	games_stats = conn.execute("""SELECT games_table.GAME_TAG, games_momentum_new.POSSESSIONS_ARR
	# 	FROM games_table
	# 		INNER JOIN games_momentum_new ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
	# 		INNER JOIN games_scores ON games_table.GAME_TAG = games_scores.GAME_TAG
	# 	""")
	# 	games_data = games_stats.fetchall()
	# 	games_count = len(games_data)
	#
	# 	games_samples_indices = np.random.choice(games_count, art_games_count, replace=False)
	# 	selected_games = [games_data[_] for _ in games_samples_indices]

	art_games_count = len(selected_games)
	insert_arr = [(None,) * 7] * art_games_count
	insert_row = [None] * 7

	insert_row[2] = delta

	for k, game in enumerate(selected_games):
		start_t = time.time()
		game_possessions_1 = game[1]
		who_start = game[1][1]

		game_pbp = str.split(game_possessions_1, '|')
		game_possessions = [(int(_) if _[0] != "'" else '0') for _ in game_pbp if not (_[0] == "'" and _[1] != "0")]  # quarter and OT filtering
		a_poss = [_ for _ in game_possessions[0::2] if type(_) is int]
		b_poss = [_ for _ in game_possessions[1::2] if type(_) is int]
		max_score = max(max(a_poss), max(b_poss))

		a_score_count = [a_poss.count(_) for _ in range(max_score + 1)]
		a_succ_prob = sum(a_score_count[1:])/sum(a_score_count)
		a_succ_given_fail_prob = a_succ_prob * (1 - delta)
		a_succ_given_succ_prob = a_succ_given_fail_prob + delta
		a_score_prob = [_/sum(a_score_count[1:]) for _ in a_score_count[1:]]

		b_score_count = [b_poss.count(_) for _ in range(max_score + 1)]
		b_succ_prob = sum(b_score_count[1:])/sum(b_score_count)
		b_succ_given_fail_prob = b_succ_prob * (1 - delta)
		b_succ_given_succ_prob = b_succ_given_fail_prob + delta
		b_score_prob = [_/sum(b_score_count[1:]) for _ in b_score_count[1:]]

		a_or_b = 0
		a_last, b_last = -1, -1
		a_score, b_score = 0, 0
		art_game = game_pbp.copy()

		for j, poss in enumerate(game_pbp):
			if poss[0] == "'":
				if poss[1] == '0':
					a_or_b = 1 - a_or_b
				continue

			if a_or_b == 0:  # a
				if a_last == -1:
					# art_game[j] = str(np.random.choice(2, p=[1 - a_succ_prob, a_succ_prob]))
					cur_score = np.random.choice(2, p=[1 - a_succ_prob, a_succ_prob])
				elif a_last == 0:
					cur_score = np.random.choice(2, p=[1 - a_succ_given_fail_prob, a_succ_given_fail_prob])
				else:
					cur_score = np.random.choice(2, p=[1 - a_succ_given_succ_prob, a_succ_given_succ_prob])

				a_last = cur_score

				if cur_score == 1:
					art_game[j] = str(np.random.choice(np.arange(1, max_score+1), p=a_score_prob))
				else:
					art_game[j] = '0'
				a_score += int(art_game[j])
			else:  # b
				if b_last == -1:
					# art_game[j] = str(np.random.choice(2, p = [1 - b_succ_prob, b_succ_prob]))
					cur_score = np.random.choice(2, p=[1 - b_succ_prob, b_succ_prob])
				elif b_last == 0:
					cur_score = np.random.choice(2, p=[1 - b_succ_given_fail_prob, b_succ_given_fail_prob])
				else:
					cur_score = np.random.choice(2, p=[1 - b_succ_given_succ_prob, b_succ_given_succ_prob])

				b_last = cur_score

				if cur_score == 1:
					art_game[j] = str(np.random.choice(np.arange(1, max_score+1), p=b_score_prob))
				else:
					art_game[j] = '0'
				b_score += int(art_game[j])

			a_or_b = 1 - a_or_b

		insert_row[0] = game[0]
		insert_row[1] = '|'.join(art_game)
		insert_row[3] = a_score if who_start == 'H' else b_score
		insert_row[4] = a_score if who_start == 'A' else b_score
		insert_row[5] = a_succ_prob if who_start == 'H' else b_succ_prob
		insert_row[6] = a_succ_prob if who_start == 'A' else b_succ_prob

		insert_arr[k] = tuple(insert_row)
		# if k % 50 == 0:
		# 	print("k={} - {}".format(k, time.time() - start_t))
		# conn.executemany("INSERT INTO artificial_games VALUES (?,?,?,?,?,?,?)", insert_arr)
		# conn.commit()
	return insert_arr


def artificial_games_area_test(games_data, test_method):

	games_count = len(games_data)

	results_arr = [(None,) * 6] * games_count
	result_row = [None] * 6

	res_calc_method = [0.] * 2000

	for k, game in enumerate(games_data):
		start_t = time.time()
		game_tag = game[0]
		game_possessions_1 = game[1]
		home_score = game[3]
		away_score = game[4]
		who_start = game[1][1]

		game_possessions_2 = [(int(_) if _[0] != "'" else '0') for _ in str.split(game_possessions_1, '|') if not (_[0] == "'" and _[1] != "0")]  # quarter and OT filtering
		# a_possessions = [_ for _ in game_possessions_2[0::2] if type(_) is int]
		# b_possessions = [_ for _ in game_possessions_2[1::2] if type(_) is int]
		a_possessions = game_possessions_2[0::2]
		b_possessions = game_possessions_2[1::2]

		a_poss_perm = GenPermReg(a_possessions)
		b_poss_perm = GenPermReg(b_possessions)

		if who_start == 'H':
			for j in range(2000):
				if test_method == 'Area':
					res_calc_method[j] = calcArea(a_poss_perm[j], b_poss_perm[j], home_score, away_score)
				elif test_method == 'Runs':
					res_calc_method[j] = calc_possessions_change(a_poss_perm[j], b_poss_perm[j])
				elif test_method == 'Max_Dist':
					res_calc_method[j] = calc_max_point_distance(a_poss_perm[j], b_poss_perm[j], home_score, away_score)

		else:
			for j in range(2000):
				if test_method == 'Area':
					res_calc_method[j] = calcArea(a_poss_perm[j], b_poss_perm[j], away_score, home_score)
				elif test_method == 'Runs':
					res_calc_method[j] = calc_possessions_change(a_poss_perm[j], b_poss_perm[j])
				elif test_method == 'Max_Dist':
					res_calc_method[j] = calc_max_point_distance(a_poss_perm[j], b_poss_perm[j], away_score, home_score)

		result_row[0] = game_tag
		result_row[1] = res_calc_method[0]
		result_row[2] = np.median(res_calc_method)
		result_row[3] = np.mean(res_calc_method)
		result_row[4] = np.var(res_calc_method, ddof=1)
		result_row[5] = sum([1 for _ in res_calc_method if _ < res_calc_method[0]]) / 2000

		results_arr[k] = tuple(result_row)

		# if k % 50 == 0:
		# 	print("k={} - {}".format(k, time.time() - start_t))

	test_result = [0.] * 4
	test_result[0] = games_count
	test_result[1] = sum([1 for _ in results_arr if _[1] >= _[2]])
	test_result[2] = sum([(_[1] - _[3])/np.sqrt(_[4]) for _ in results_arr])/np.sqrt(games_count)
	test_result[3] = np.average([_[5] for _ in results_arr])

	return test_result


def power_analysis_multi_thread(games_data, realizations_indicies, delta, test_method, art_games_count, proc_name, queue):

	games_count = len(games_data)
	num_of_realizations = len(realizations_indicies)

	insert_arr = [(None,) * 7] * num_of_realizations
	insert_row = [None] * 7

	for k, i in enumerate(realizations_indicies):
		start_t = time.time()
		games_samples_indices = np.random.choice(games_count, art_games_count, replace=False)
		selected_games = [games_data[_] for _ in games_samples_indices]
		art_games = gen_artificial_games(selected_games, delta)

		insert_row[0] = test_method
		insert_row[1] = delta
		insert_row[2] = i
		insert_row[3:] = artificial_games_area_test(art_games, test_method)
		insert_arr[k] = tuple(insert_row)

		print("{}: k = {},\ti = {},\ttime = {}".format(proc_name, k+1, i, time.time() - start_t))

	queue.put(insert_arr)


def power_analysis_main_thread(num_of_realizations, delta, test_method, art_games_count):

	with sqlite3.connect('Games.db') as conn:

		# # conn.execute("DROP TABLE IF EXISTS power_analysis")
		conn.execute("""
		CREATE TABLE IF NOT EXISTS power_analysis(
			TEST_METHOD TEXT NOT NULL,
			DELTA REAL NOT NULL,
			TEST_NUM INT NOT NULL,
			ART_GAMES_COUNT INT NOT NULL,
			BIGGER_THAN_MEDIAN INT NOT NULL,
			TEST_Z_SCORE REAL NOT NULL,
			P_VALUE_AVG REAL NOT NULL
		);
		""")
		games_stats = conn.execute("""SELECT games_table.GAME_TAG, games_momentum_new.POSSESSIONS_ARR
		FROM games_table
			INNER JOIN games_momentum_new ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
			INNER JOIN games_scores ON games_table.GAME_TAG = games_scores.GAME_TAG
		""")
		games_data = games_stats.fetchall()
		games_count = len(games_data)
		insert_arr = [(None,) * 7] * num_of_realizations

		res_queue = mp.Queue()
		num_of_thread = 2
		process_arr = [None] * num_of_thread
		for i in range(num_of_thread):
			process_arr[i] = mp.Process(target=power_analysis_multi_thread, args=(games_data, list(range(751 + i*(num_of_realizations//num_of_thread), 751 + (i+1)*(num_of_realizations//num_of_thread))), delta, test_method, art_games_count, "Prc_{}".format(i+1), res_queue))
			process_arr[i].start()

		k = 0

		for i in range(num_of_thread):
			cur_res = res_queue.get()

			for test_res in cur_res:
				insert_arr[k] = test_res
				k += 1

		for i in range(num_of_thread):
			process_arr[i].join()

		conn.executemany("INSERT INTO power_analysis VALUES (?,?,?,?,?,?,?)", insert_arr)
		conn.commit()
		print(insert_arr)


def calc_delta_games():
	with sqlite3.connect('Games.db') as conn:

		games_stats = conn.execute("""SELECT games_table.GAME_TAG, games_momentum_new.POSSESSIONS_ARR
		FROM games_table
			INNER JOIN games_momentum_new ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
		WHERE SEASON BETWEEN 2002 AND 2017
		""")
		games_data = games_stats.fetchall()

		pbp_arr = [_[1] for _ in games_data]

		delta_arr = [0.] * len(pbp_arr)
		print(len(pbp_arr))

		for k, poss_arr in enumerate(pbp_arr):
			game_possessions_2 = [(int(_) if _[0] != "'" else '0') for _ in str.split(poss_arr, '|') if not (_[0] == "'" and _[1] != "0")]  # quarter and OT filtering
			# a_possessions = [_ for _ in game_possessions_2[0::2] if type(_) is int]
			# b_possessions = [_ for _ in game_possessions_2[1::2] if type(_) is int]
			a_possessions = game_possessions_2[0::2]
			b_possessions = game_possessions_2[1::2]

			play_after_success_count, play_after_loss_count = 0, 0
			success_after_success_count, success_after_loss_count = 0, 0

			last_a = -1
			last_b = -1

			for i, poss in enumerate(a_possessions):
				if poss == '0':
					continue

				if last_a == 0:
					play_after_loss_count += 1
					if poss > 0:
						success_after_loss_count += 1
				elif last_a == 1:
					play_after_success_count += 1
					if poss > 0:
						success_after_success_count += 1

				last_a = 1 if poss > 0 else 0

			for i, poss in enumerate(b_possessions):
				if poss == '0':
					continue

				if last_b == 0:
					play_after_loss_count += 1
					if poss > 0:
						success_after_loss_count += 1
				elif last_b == 1:
					play_after_success_count += 1
					if poss > 0:
						success_after_success_count += 1

				last_b = 1 if poss > 0 else 0

			delta_arr[k] = (success_after_success_count/play_after_success_count) - (success_after_loss_count/play_after_loss_count)

		plt.figure()
		plt.hist(delta_arr, bins=100)
		plt.show()


def calc_changes_median_prob(start_season, end_season):
	with sqlite3.connect('Games.db') as conn:
		# conn.execute("ALTER TABLE games_changes_momentum2 ADD COLUMN MEDIAN_LESS INT;")
		# conn.execute("ALTER TABLE games_changes_momentum2 ADD COLUMN MEDIAN_EQUAL INT;")
		# conn.commit()
		# conn.execute("DROP TABLE IF EXISTS games_changes_momentum2")
		# conn.execute("""
		# CREATE TABLE IF NOT EXISTS games_changes_momentum2(
		# 	GAME_TAG TEXT NOT NULL UNIQUE,
		# 	CHANGES_COUNT INT NOT NULL,
		# 	MEDIAN REAL NOT NULL,
		# 	MEAN REAL NOT NULL,
		# 	VAR REAL NOT NULL,
		# 	P_VALUE REAL NOT NULL
		# );
		# """)
		insert_arr = [(None,) * 3] * 2000  # 2000 - max number of games per season
		insert_row = [None] * 3
		res_changes = [0] * 8000
		row_index = 0

		for i in range(start_season, end_season - 1, -1):
			result = conn.execute("""
			SELECT COUNT(*)
			FROM games_momentum_new
				INNER JOIN games_table ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
			WHERE SEASON=?
			""", (i,))
			num_of_rows = result.fetchone()[0]
			games_stats = conn.execute("""SELECT games_table.GAME_TAG, games_momentum_new.POSSESSIONS_ARR, games_table.HOME_SCORE, games_table.AWAY_SCORE
			FROM games_momentum_new
				INNER JOIN games_table ON games_momentum_new.GAME_TAG = games_table.GAME_TAG
			WHERE SEASON=?
			""", (i,))

			games_data = games_stats.fetchall()
			row_index = 0

			for k, game in enumerate(games_data):
				start_t = time.time()
				game_tag = game[0]
				game_possessions_1 = game[1]
				home_score = game[2]
				away_score = game[3]
				who_start = game[1][1]

				game_possessions_2 = [(int(_) if _[0] != "'" else '0') for _ in str.split(game_possessions_1, '|') if not (_[0] == "'" and _[1] != "0")]  # quarter and OT filtering
				# a_possessions = [_ for _ in game_possessions_2[0::2] if type(_) is int]
				# b_possessions = [_ for _ in game_possessions_2[1::2] if type(_) is int]
				a_possessions = game_possessions_2[0::2]
				b_possessions = game_possessions_2[1::2]

				a_poss_perm = GenPermReg(a_possessions)
				b_poss_perm = GenPermReg(b_possessions)

				if who_start == 'H':
					for j in range(8000):
						res_changes[j] = calc_possessions_change(a_poss_perm[j], b_poss_perm[j])
				else:
					for j in range(8000):
						res_changes[j] = calc_possessions_change(a_poss_perm[j], b_poss_perm[j])

				# insert_row[0] = game_tag
				# insert_row[1] = res_changes[0]
				# insert_row[2] = np.median(res_changes)
				# insert_row[3] = np.mean(res_changes)
				# insert_row[4] = np.var(res_changes, ddof=1)
				# insert_row[5] = sum([1 for _ in res_changes if _ < res_changes[0]]) / 8000
				insert_row[2] = game_tag
				game_median = np.median(res_changes)
				insert_row[0] = sum([1 for _ in res_changes if _ > game_median])
				insert_row[1] = sum([1 for _ in res_changes if _ == game_median])

				insert_arr[row_index] = tuple(insert_row)

				row_index += 1

				if k % 50 == 0:
					print("k={} - {}".format(k, time.time() - start_t))

			conn.executemany("""UPDATE games_changes_momentum2 SET MEDIAN_LESS=?, MEDIAN_EQUAL=? WHERE GAME_TAG = ?""", insert_arr[0:row_index])
			conn.commit()
			print("finished season {}".format(i))


def data_base_transfer():
	with sqlite3.connect('Games.db') as conn1, sqlite3.connect('Games3.db') as conn2:

		result = conn2.execute("""
		SELECT
		*
		FROM power_analysis
		WHERE power_analysis.DELTA = 0.06 AND power_analysis.TEST_METHOD = 'Runs'
		""")

		result_data = result.fetchall()

		conn1.executemany("INSERT INTO power_analysis VALUES (?,?,?,?,?,?,?)", result_data)
		conn1.commit()


if __name__ == '__main__':
	matplotlib.rcParams.update({'font.size': 20, 'figure.figsize': (14, 8), 'figure.autolayout': True})

	# start_time = time.time()
	# calc_momentum()
	# # check_res('200710310DEN', 2008, 120, 103)
	# print("finished: ", time.time() - start_time)
	# print(NBA_Parser.pbp_to_possessions('Games/2013-2014/201401070MIL.csv'))
	# c = scipy.stats.binom_test(15019, 21291, 0.5)
	# print(c)
	# print(scipy.stats.binom.sf(np.float64(5), 10, 0.5))

	# area_hist()
	# plot_graphs()
	# calc_global_momentum()
	# games_possessions_change()
	# area_hist()
	# area_hist_poss_changes()
	# area_hist_poss_max_dist()
	# area_hist_poss_global()
	# csv_to_possessions()
	# check_possessions()
	# calc_area_h_a(2018, 2002)
	# calc_changes_h_a(2018, 2002)
	# calc_possessions_dist(2018, 2002)
	# calc_global_momentum()
	# check_bias()
	# calc_area_corr()
	plot_graphs()
	# check_possessions()
	# gen_artificial_games(1000, 0)
	# plot_pbp_game("'H'|'1'|2|2|0|0|2|2|3|0|3|0|2|2|2|0|0|0|2|2|2|0|0|3|2|0|2|2|0|0|2|0|2|0|3|0|2|0|3|0|2|2|0|0|2|0|2|3|0|'2'|2|0|3|0|0|0|0|2|0|2|3|2|0|0|0|3|0|0|2|2|1|0|0|2|3|2|0|2|0|2|1|0|2|2|0|0|2|2|0|2|2|0|0|2|0|0|0|0|0|0|'3'|0|0|1|2|2|0|0|2|2|3|0|2|2|0|2|3|0|2|0|2|2|0|2|0|2|0|2|0|0|0|0|3|2|0|0|0|2|2|3|0|3|0|3|0|0|0|2|0|2|0|0|'4'|2|2|0|0|0|0|0|2|0|2|0|2|2|0|0|0|0|3|2|3|0|2|2|2|0|0|0|2|0|0|2|0|0|2|0|2|0|0|1", 'H', 99, 97, draw_linear=True, draw_area=True)
	# plot_pbp_game("'H'|'1'|2|0|0|2|0|2|0|2|2|0|0|3|2|0|3|0|0|0|0|0|2|0|0|2|0|0|0|2|2|2|2|2|0|0|0|2|2|0|2|2|0|0|0|0|'2'|'0'|0|2|2|2|0|0|0|1|0|2|2|2|0|0|0|2|2|0|3|2|2|2|2|2|0|2|2|0|2|2|2|3|2|2|1|0|0|2|4|2|'3'|0|0|0|0|2|2|0|0|0|0|0|0|2|2|2|0|2|1|0|3|3|0|0|2|3|2|2|0|0|2|0|3|2|2|0|0|0|2|0|2|2|'4'|2|2|0|2|3|2|0|2|0|2|3|2|2|2|3|0|2|3|2|0|3|0|0|2|1|0|0|3|0|0|0|3|2|0|2|2|0|0|2|3", 'H', 99, 97, draw_linear=True, draw_area=True)
	# plot_pbp_game("'H'|'1'|0|0|0|2|0|3|0|0|0|2|2|0|2|0|0|2|0|0|0|2|2|2|2|2|2|0|2|0|3|2|2|0|0|1|0|2|0|2|2|3|0|0|2|3|0|3|'2'|'0'|2|2|0|2|0|0|0|2|0|0|0|2|0|2|3|2|0|2|3|2|0|2|0|2|0|0|2|0|0|0|2|2|0|0|0|0|4|0|2|2|0|'3'|'0'|2|0|2|0|2|2|0|0|0|2|2|0|3|0|0|0|0|2|0|3|2|0|3|1|0|0|2|0|1|2|3|3|0|0|2|2|2|0|0|2|0|1|2|3|'4'|'0'|0|0|3|0|3|2|2|3|0|2|1|0|2|3|3|2|0|2|0|0|2|0|3|0|2|0|2|1|2|0|2|3|0|2|2|0|2|0", 'H', 99, 97, draw_area=True, draw_linear=True, block=True)
	# plot_pbp_game("'H'|'1'|0|0|0|2|0|3|0|0|0|2|2|0|2|0|0|2|0|0|0|2|2|2|2|2|2|0|2|0|3|2|2|0|0|1|0|2|0|2|2|3|0|0|2|3|0|3|'2'|'0'|2|2|0|2|0|0|0|2|0|0|0|2|0|2|3|2|0|2|3|2|0|2|0|2|0|0|2|0|0|0|2|2|0|0|0|0|4|0|2|2|0|'3'|'0'|2|0|2|0|2|2|0|0|0|2|2|0|3|0|0|0|0|2|0|3|2|0|3|1|0|0|2|0|1|2|3|3|0|0|2|2|2|0|0|2|0|1|2|3|'4'|'0'|0|0|3|0|3|2|2|3|0|2|1|0|2|3|3|2|0|2|0|0|2|0|3|0|2|0|2|1|2|0|2|3|0|2|2|0|2|0",
	# 'H', 99, 97, draw_area=True, draw_linear=True, block=False)
	# plot_pbp_game("'H'|'1'|2|2|0|0|2|2|3|0|3|0|2|2|2|0|0|0|2|2|2|0|0|3|2|0|2|2|0|0|2|0|2|0|3|0|2|0|3|0|2|2|0|0|2|0|2|3|0|'2'|2|0|3|0|0|0|0|2|0|2|3|2|0|0|0|3|0|0|2|2|1|0|0|2|3|2|0|2|0|2|1|0|2|2|0|0|2|2|0|2|2|0|0|2|0|0|0|0|0|0|'3'|0|0|1|2|2|0|0|2|2|3|0|2|2|0|2|3|0|2|0|2|2|0|2|0|2|0|2|0|0|0|0|3|2|0|0|0|2|2|3|0|3|0|3|0|0|0|2|0|2|0|0|'4'|2|2|0|0|0|0|0|2|0|2|0|2|2|0|0|0|0|3|2|3|0|2|2|2|0|0|0|2|0|0|2|0|0|2|0|2|0|0|1",
	# 'H', 99, 97, draw_linear=False, draw_area=False, new_figure=False, draw_legend=False, block=True)
	# plot_pbp_game("'H'|'1'|2|0|2|0|2|3|2|0|2|0|0|0|0|2|0|0|0|0|0|2|2|0|3|0|0|3|2|0|0|0|0|0|2|3|0|2|2|2|0|0|0|0|2|0|2|2|3|0|2|0|2|2|1|0|'2'|'0'|0|0|2|0|3|0|0|3|0|0|3|0|3|0|0|2|0|2|0|0|0|0|3|2|0|0|0|2|0|2|2|0|0|2|0|3|2|2|2|0|0|0|2|0|2|2|3|2|2|2|2|0|'3'|2|3|3|2|3|0|0|2|0|3|0|1|0|0|3|2|2|0|2|0|3|0|0|0|0|2|0|2|0|2|3|2|2|0|2|0|0|2|0|2|0|2|2|3|2|3|2|0|2|2|0|'4'|0|2|0|0|0|3|0|2|0|2|0|0|2|0|0|0|0|3|2|2|0|0|2|0|2|2|0|0|1|0|2|2|0|2|0|0|2|2|3|3|0|0|2|0|2|2|0|0|1|3|0|'1OT'|'0'|2|2|0|0|0|0|2|1|0|0|0|0|3|0|3|0|2|3|1|0|2|0", 'H', 130, 121, draw_linear=True, draw_area=True, block=False)
	# plot_pbp_game("'A'|'1'|0|0|0|0|0|0|0|2|0|2|3|0|3|2|2|0|2|0|3|2|0|0|0|2|0|3|2|0|0|0|0|2|2|0|2|2|3|2|2|0|3|0|3|0|3|1|3|2|3|'2'|3|0|0|0|0|2|2|0|2|0|0|2|2|0|0|0|0|0|3|3|0|0|2|2|3|0|2|2|2|0|2|0|0|3|2|2|2|3|0|0|0|3|0|0|2|0|0|0|'3'|3|0|2|0|0|0|3|0|0|0|3|2|0|0|2|3|0|0|2|3|3|2|1|3|0|0|0|0|2|0|1|0|3|0|3|0|3|2|0|0|0|0|2|1|'4'|'0'|0|0|0|0|0|2|0|3|2|2|0|3|0|0|0|0|0|2|0|3|0|2|2|0|0|2|0|3|0|3|0|2|2|0|3|0|0|0|0|2|0|2|0", 'A', 115, 86, draw_area=True, draw_linear=True, block=False)
	# plot_pbp_game("'H'|'1'|3|2|0|0|2|2|0|0|2|0|2|0|3|0|3|2|3|3|2|2|0|0|0|0|0|2|2|0|0|3|0|0|0|0|2|0|0|2|2|0|2|2|2|0|0|0|'2'|'0'|2|2|2|0|2|0|0|0|0|0|2|2|2|0|2|0|1|2|0|0|0|2|2|0|2|2|0|2|0|3|0|2|2|2|3|2|3|2|0|0|2|0|3|0|0|0|'3'|2|2|2|3|0|0|0|0|2|2|2|2|0|0|3|0|0|2|2|0|2|0|0|2|2|0|0|0|0|0|0|3|0|0|0|0|0|2|2|1|2|2|0|2|2|2|'4'|'0'|0|2|0|0|0|3|2|2|0|1|0|3|3|0|0|2|0|2|2|2|0|2|0|0|3|2|0|2|2|2|0|0|0|2|0|2|2|0|0", 'H', 92, 102, draw_area=True, draw_linear=True, block=True)
	# calc_ot_winning_prob()
	# calc_delta_games()
	# power_analysis_main_thread(250, 0.03, "Runs", 1000)
	# time.sleep(30)
	# power_analysis_main_thread(250, 0.01, "Runs", 1000)
	# area_hist_poss_global()
	# calc_changes_median_prob(2018, 2002)
