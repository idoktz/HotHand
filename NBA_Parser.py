import numpy as np
import array
import time
import re
import urllib.request, urllib.error # .request
from bs4 import BeautifulSoup, Comment
import csv
import os
import sqlite3
from itertools import islice
from datetime import datetime
import sys

time_var = 0
EMA_var = 0


def Playoffs_Parser(base_url):

	with sqlite3.connect('Games.db') as conn:  # open('Playoffs_list_ByYear.csv', 'w', newline='') as csvfile, sqlite3.connect('Games.db') as conn:
		#  conn.execute("DROP TABLE IF EXISTS games_table")
		# conn.execute('''CREATE TABLE IF NOT EXISTS games_table(
		# GAME_TAG TEXT NOT NULL PRIMARY KEY,
		# SEASON INT,
		# YEAR INT,
		# MONTH INT,
		# DAY INT,
		# HOME_TEAM TEXT NOT NULL,
		# AWAY_TEAM TEXT NOT NULL,
		# HOME_SCORE INT NOT NULL,
		# AWAY_SCORE INT NOT NULL,
		# OT INT,
		# PBP_URL TEXT NOT NULL
		# );
		# ''')
		insert_rows = [(None,)*11] * 2000
		row_num = 0
		row_values = [None] * 11
		# game_tag_regex = re.compile("(?:.*)/([^/]+)\.html")
		# ot_regex = re.compile("(\d*)OT")

		# writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
		# dict_writer = csv.DictWriter(csvfile, ['GAME_TAG', 'Year', 'Month', 'Day', 'Home Name', 'Away Name', 'Home score', 'Away score', 'OT', 'PbP URL'])
		# dict_writer.writeheader()
		global time_var

		for i in range(2018, 2017, -1):
			cur_url = base_url + '/leagues/NBA_' + str(i) + '_games.html'  # https://www.basketball-reference.com/leagues/NBA_2017_games.html
			start_t = time.time()
			r = urllib.request.urlopen(cur_url)
			html = r.read()
			time_var += time.time() - start_t
			soup = BeautifulSoup(html, "html5lib")
			
			months_list = soup.find("div", id='content').find("div", attrs={'class': 'filter'}).find_all('a', href=re.compile("leagues/NBA_\d+_games-.+\.html"))

			row_values[1] = i
			for month in months_list:
				start_t = time.time()
				soup2 = BeautifulSoup(urllib.request.urlopen(base_url + month.get('href')).read(), "html5lib")
				time_var += time.time() - start_t
				games_table = soup2.find("table", id="schedule").find("tbody").find_all("tr")  # , attrs={"class": re.compile("toggleable.*$")})

				for game_row in games_table:
					cols = game_row.find_all(['th', 'td'])
					if len(cols) < 10:
						continue
					row_values[0] = game_tag_regex.search(base_url + cols[6].a.get('href')).group(1)
					# tmp_date = re.split("(?: |,)+", cols[0].text)
					tmp_date = datetime.strptime(cols[0].text, '%a, %b %d, %Y')
					row_values[2:5] = [tmp_date.year, tmp_date.month, tmp_date.day]
					row_values[5:7] = [cols[4].text, cols[2].text]
					row_values[7:9] = [int(cols[5].text), int(cols[3].text)]
					ot_num = ot_regex.search(cols[7].text)
					if ot_num is not None:
						row_values[9] = int(ot_num.group(1)) if ot_num.group(1) != '' else 1
					else:
						row_values[9] = None
					row_values[10] = 'https://www.basketball-reference.com/boxscores/pbp/' + row_values[0] + '.html'
					insert_rows[row_num] = tuple(row_values)
					row_num += 1
					# insert_rows.append(tuple(row_values))
					# prev_tr = game_row.previous_sibling.previous_sibling
					# prev_td = prev_tr.td.next_sibling.next_sibling
					# team_names = [a_text.text for a_text in game.find_all("a", href=re.compile("teams"), limit=2)]
					# games_links = game_row.find_all("a", href=re.compile("/boxscores/[0-9]+.*$"))
					# games_urls = [base_url + "/boxscores/pbp/" + str.split(a_text['href'], "/")[2] for a_text in games_links]
					# dict_writer.writerow({'Year': i, 'Team1 Name': team_names[0], 'Team2 Name': team_names[1], 'URLs': ' '.join(games_urls)})

				# writer.writerow([])
			conn.executemany("INSERT INTO games_table VALUES (?,?,?,?,?,?,?,?,?,?,?)", insert_rows[0:row_num])
			row_num = 0
			print("finished season {}-{}".format(i-1, i))
		conn.commit()


def playoffs_series_parser(base_url):
	with sqlite3.connect('Games.db') as conn:  # open('Playoffs_list_ByYear.csv', 'w', newline='') as csvfile, sqlite3.connect('Games.db') as conn:
		conn.execute("DROP TABLE IF EXISTS series_table")
		conn.execute("""CREATE TABLE IF NOT EXISTS series_table(
							GAME_TAG TEXT NOT NULL PRIMARY KEY,
							LEAGUE TEXT NOT NULL,
							SEASON INT,
							SERIES_ID INT,
							GAME_NUM INT,
							HOME_TEAM TEXT NOT NULL,
							AWAY_TEAM TEXT NOT NULL,
							HOME_SCORE INT NOT NULL,
							AWAY_SCORE INT NOT NULL
		);
		""")
		insert_rows = [(None,) * 9] * 2000
		row_values = [None] * 9
		row_num = 0
		# game_tag_regex = re.compile("(?:.*)/([^/]+)\.html")
		# ot_regex = re.compile("(\d*)OT")

		# writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
		# dict_writer = csv.DictWriter(csvfile, ['GAME_TAG', 'Year', 'Month', 'Day', 'Home Name', 'Away Name', 'Home score', 'Away score', 'OT', 'PbP URL'])
		# dict_writer.writeheader()

		for i in range(2017, 1946, -1):
			start_t = time.time()
			for league_name in ['NBA', 'ABA', 'BAA']:
				cur_url = base_url + '/playoffs/' + league_name + '_' + str(i) + '.html'  # ex: https://www.basketball-reference.com/playoffs/NBA_1965.html
				try:
					r = urllib.request.urlopen(cur_url)
				except urllib.error.HTTPError as err:
					if err.code == 404:
						continue
				html = r.read()
				soup = BeautifulSoup(html, "html5lib")

				series_table = soup.find("table", id='all_playoffs')  # .find("div", attrs={'class': 'filter'}).find_all('a', href=re.compile("leagues/NBA_\d+_games-.+\.html"))
				if series_table is None:
					print(i, league_name)
					continue

				series_list = series_table.find("tbody").find_all("tr", id=re.compile("s(\d+)"))
				row_values[1] = league_name
				row_values[2] = i
				for series in series_list:
					row_values[3] = int(series.get('id')[1:])
					games_table = series.find("table").find("tbody").find_all("tr")  # , attrs={"class": re.compile("toggleable.*$")})

					for game_row in games_table:
						cols = game_row.find_all(['th', 'td'])
						if len(cols) < 6:
							continue
						row_values[0] = game_tag_regex.search(base_url + cols[0].a.get('href')).group(1)
						row_values[4] = int(cols[0].text[5:])
						row_values[5:7] = [cols[4].text[2:], cols[2].text]
						row_values[7:9] = [int(cols[5].text), int(cols[3].text)]
						insert_rows[row_num] = tuple(row_values)
						row_num += 1
					# insert_rows.append(tuple(row_values))
					# prev_tr = game_row.previous_sibling.previous_sibling
					# prev_td = prev_tr.td.next_sibling.next_sibling
					# team_names = [a_text.text for a_text in game.find_all("a", href=re.compile("teams"), limit=2)]
					# games_links = game_row.find_all("a", href=re.compile("/boxscores/[0-9]+.*$"))
					# games_urls = [base_url + "/boxscores/pbp/" + str.split(a_text['href'], "/")[2] for a_text in games_links]
					# dict_writer.writerow({'Year': i, 'Team1 Name': team_names[0], 'Team2 Name': team_names[1], 'URLs': ' '.join(games_urls)})

					# writer.writerow([])
			conn.executemany("INSERT INTO series_table VALUES (?,?,?,?,?,?,?,?,?)", insert_rows[0:row_num])
			row_num = 0
			print("finished season {} in {} seconds".format(i, time.time() - start_t))
		conn.commit()


def games_quarter_scores(base_url):
	with sqlite3.connect('Games.db') as conn:
		# conn.execute("DROP TABLE IF EXISTS games_scores")
		# conn.execute("""
		# CREATE TABLE IF NOT EXISTS games_scores(
		# 	GAME_TAG TEXT NOT NULL PRIMARY KEY,
		# 	HOME_SCORE INT NOT NULL,
		# 	AWAY_SCORE INT NOT NULL,
		# 	QT1_H_A TEXT,
		# 	QT2_H_A TEXT,
		# 	QT3_H_A TEXT,
		# 	QT4_H_A TEXT,
		# 	OT1_H_A TEXT,
		# 	OT2_H_A TEXT,
		# 	OT3_H_A TEXT,
		# 	OT4_H_A TEXT
		# );
		# """)
		insert_rows = [(None,) * 14] * 2000
		row_values = [None] * 14

		for i in range(2018, 2017, -1):
			result = conn.execute("""
			SELECT COUNT(*)
			FROM games_table
			WHERE SEASON=?
			""", (i,))
			num_of_rows = result.fetchone()[0]
			games_stats = conn.execute("""
			SELECT games_table.GAME_TAG, games_table.OT
			FROM games_table
			WHERE SEASON=?
			""", (i,))

			games_data = games_stats.fetchall()
			row_index = 0

			for k, game in enumerate(games_data):
				start_t = time.time()
				game_tag = game[0]
				ot_count = game[1] if game[1] is not None else 0

				try:
					cur_url = base_url + "/boxscores/" + game_tag + ".html"  # https://www.basketball-reference.com/boxscores/201611230PHI.html
					r = urllib.request.urlopen(cur_url)
				except urllib.error.HTTPError as err:
					if err.code == 404:
						print("Error:\t", game_tag)
						continue
				html = r.read(70000).decode('utf-8')
				table_search = re.search("<table.{0,50}?id=\"line_score\".{0,50}?>.+?</table>", html, flags=re.I | re.DOTALL)
				table_text = html[table_search.start():table_search.end()]
				# soup = BeautifulSoup(html, "html5lib")
				# scores_table2 = BeautifulSoup(str(soup.find(text=lambda text: isinstance(text, Comment) and re.search("<table .+ id=\"line_score\" .+>", text))), "html5lib")
				scores_table2 = BeautifulSoup(table_text, "html5lib")
				# print(scores_table2.find("table", id='line_score'))
				# print(scores_table2.find("table", id='line_score').find("tbody"))
				scores_table = scores_table2.find("table", id='line_score').find("tbody").find_all("tr")
				away_scores = scores_table[2].find_all("td")
				home_scores = scores_table[3].find_all("td")

				if len(away_scores) - 6 != ot_count or len(home_scores) - 6 != ot_count:
					print("OT count error:\t", game_tag, len(away_scores), ot_count)
					continue

				row_values[0] = game_tag
				row_values[1], row_values[2] = int(home_scores[-1].text), int(away_scores[-1].text)

				for j in range(8):
					if j >= 4 + ot_count:
						row_values[j + 3] = None
					else:
						row_values[j + 3] = "{}-{}".format(home_scores[j+1].text, away_scores[j+1].text)

				insert_rows[row_index] = tuple(row_values)
				row_index += 1
				start_t = time.time() - start_t
				if k%100 == 0:
					print("{} - {}".format(k, start_t))

			conn.executemany("INSERT INTO games_scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", insert_rows[0:row_index])
			row_index = 0
			print("finished season {} in {} seconds avg".format(i, time.time() - start_t))

		conn.commit()


def Game_Parser(game_id, game_season, game_url):

	dir_name = "Games/{}-{}/".format(game_season-1, game_season)
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)

	#  game_id = str.split(game_url, '/')[5].split('.')[0]
	#  game_id = game_tag_regex.search(game_url).group(1)
	
	start_t = time.time()
	for k in range(5):
		try:
			r = urllib.request.urlopen(game_url)
			html = r.read()
		except:
			print("\rfailed urlopen {} out of 5".format(k))
			if k == 4:
				raise
			time.sleep(10)
		else:
			break
	global time_var
	global EMA_var
	EMA_var = time.time() - start_t
	time_var += EMA_var
	soup = BeautifulSoup(html, "html5lib")
	
	title = names_regex.search(soup.find("div", id="content").find("h1").string)
	home_name = title.group(2)
	away_name = title.group(1)

	pbp_table = soup.find("table", id="pbp").find("tbody").find_all("tr")

	with open(dir_name + game_id + ".csv", 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
		writer.writerow([title.group(1), title.group(2), title.group(3), title.group(4)])
		writer.writerow('')
		writer.writerow(['Time', 'H/A action', 'Home: ' + home_name, 'Away: ' + away_name, 'score change', 'description'])
		arr2csv = [''] * 6
		
		for table_row in pbp_table:
			row_cells = table_row.find_all("td")

			if len(row_cells) == 2:
				arr2csv[0] = row_cells[0].get_text()
				arr2csv[1] = row_cells[1].get_text()
				writer.writerow([arr2csv[0], '-', arr2csv[1]])

			elif len(row_cells) == 6:
				arr2csv[0] = row_cells[0].get_text()
				is_home_action = (row_cells[1].get_text() == '\xa0')
				cur_score = row_cells[3].get_text().split('-')
				arr2csv[1] = 'H' if is_home_action else 'A'
				arr2csv[2] = cur_score[1]
				arr2csv[3] = cur_score[0]
				arr2csv[4] = row_cells[4 if is_home_action else 2].get_text()
				arr2csv[5] = row_cells[5 if is_home_action else 1].get_text()
				writer.writerow(arr2csv)


def Games_Parser():

	global EMA_var
	EMA_time = 0

	with sqlite3.connect('Games.db') as conn:

		for i in range(2018, 2017, -1):
			result = conn.execute("SELECT COUNT(*) FROM games_table WHERE SEASON=?", (i,))
			num_of_rows = result.fetchone()[0]
			games_list = conn.execute("SELECT GAME_TAG, PBP_URL FROM games_table WHERE SEASON=?", (i,))

			for index, game in enumerate(games_list, start=1):
				Game_Parser(game[0], i, game[1])
				EMA_time = (1/6)*EMA_time + (5/6)*EMA_var
				sys.stdout.write('\r')
				if (index % (num_of_rows//10)) == 0:
					sys.stdout.write("sleep.\tindex={},\tnum_of_rows={}".format(index, num_of_rows))
					time.sleep(10)  # sleep for 10 seconds
				else:
					sys.stdout.write("{}% finished ({} avg sec for game:\t{} min remains)".format(index/num_of_rows*100, EMA_time, EMA_time*(num_of_rows - index)/60))
				sys.stdout.flush()

			print("\nFinished season {}-{}".format(i-1, i))


names_regex = re.compile("(.+) at (.+) Play-By-Play, (.+), (.+)$")
game_tag_regex = re.compile("(?:.*)/([^/]+)\.html")
ot_regex = re.compile("(\d*)OT")
game_tag_from_file_name = re.compile("/?(?:.*/)?(.+)\.csv")
start_of_nth_q_regex = re.compile("Start of (\d+)(\S+) (quarter|overtime)", flags=re.I)
jump_ball_regex = re.compile("Jump ball: (.+) vs\. (.+)(?: \((.+)\))?", flags=re.I)
game_pt_regex = re.compile("(.+) (makes|misses) .*(\d)-pt shot (.+)(?:\( assist by (.+)\))?", flags=re.I)
free_throw_regex = re.compile("(.+) (makes|misses) .*free throw(?: (.*)(?:(\d) of (\d))?)?", flags=re.I)
rebound_regex = re.compile("(Defensive|Offensive) rebound by (.+)", flags=re.I)
foul_regex = re.compile("(.*) ?foul by (.+)(?: \(drawn by (.+)\))?", flags=re.I)
turnover_regex = re.compile("Turnover by (.+)(?: \((.+)\))", flags=re.I)
timeout_regex = re.compile("(.+) timeout.*", flags=re.I)
player_substitution_regex = re.compile("(.+) enters the game for (.+)", flags=re.I)


def pbp_to_possessions(game_file):

	with open(game_file, "r", newline='') as csvfile:  # , sqlite3.connect('Games.db') as conn:
		reader = csv.reader(csvfile)
		# conn.execute("DROP TABLE IF EXISTS pbp_table")
		# db_cursor = conn.cursor()
		# db_cursor.execute('''
		# CREATE TABLE IF NOT EXISTS pbp_table(
		# GAME_TAG TEXT NOT NULL,
		# HOME_SCORE INT,
		# AWAY_SCORE INT,
		# TIME_MIN INT,
		# TIME_SEC INT,
		# TIME_MILLI INT,
		# QUARTER TEXT,
		# HorA_ACTION TEXT NOT NULL,
		# SCORE_CHANGE INT,
		# DESCRIPTION TEXT
		# );
		# ''')

		# insert_arr = [(None, ) * 10] * 1000  # max number of rows to insert
		possessions_arr = ["-"] + [0] * 1000  # max number of possessions
		row_values = [None] * 10  # [None] * 10
		row_num = 0

		possessions_index = 1
		cur_time = 0
		cur_quarter = '1'
		last_writen_possession = ''
		score_A, score_H = 0, 0
		team_complement = {'H': 'A', 'A': 'H'}  # used to turn possession

		row_values[0] = game_tag_from_file_name.search(game_file).group(1)  # GAME TAG

		for row in islice(reader, 3, None):
			cur_time = re.split("\D+", row[0])
			row_values[3:6] = map(int, cur_time)  # TIME
			row_values[6] = cur_quarter  # QUARTER
			row_values[8] = None

			if row[1] != '-':  # team action
				row_values[7] = row[1]  # HorA_ACTION
				row_values[9] = row[5]  # DESCRIPTION
				score_A, score_B = int(row[2]), int(row[3])
				row_values[1], row_values[2] = row[2], row[3]  # TEAM SCORES

				res = game_pt_regex.search(row[5])
				if res is not None:  # game point
					if last_writen_possession == row[1]:
						possessions_index -= 1
					elif last_writen_possession == row[1] + ' ':
						possessions_arr[possessions_index] = '0'  # place holder
						possessions_index += 1

					if res.group(2) == 'makes':
						possessions_arr[possessions_index] += int(row[4])  # append new team possession's score
						possessions_index += 1
						last_writen_possession = row[1]
						row_values[8] = int(row[4])  # SCORE CHANGE
					else:  # misses
						# possessions_arr[possessions_index] += 0  # append new team possession
						possessions_index += 1
						last_writen_possession = row[1]

				else:
					res = free_throw_regex.search(row[5])
					if res is not None:  # free throw
						# if res.group(3) is not None and res.group(4) is not None:
						# 	free_throws_count = int(res.group(4))
						# 	cur_free_throw = int(res.group(3))
						#
						# 	if last_writen_possession != row[1] and cur_free_throw > 1:
						# 		print(" {}: Error in game possessions (free throw)".format(row_values[0]))
						# 		raise Exception('Error in possessions')

						if last_writen_possession == row[1]:
							possessions_index -= 1
						elif last_writen_possession == row[1] + ' ':
							possessions_arr[possessions_index] = '0'  # place holder
							possessions_index += 1

						if res.group(2) == 'makes':
							possessions_arr[possessions_index] += int(row[4])  # append new team possession's score
							possessions_index += 1
							last_writen_possession = row[1]
							row_values[8] = int(row[4])  # SCORE CHANGE
						else:  # misses
							# possessions_arr[possessions_index] += 0  # append new team possession
							possessions_index += 1
							last_writen_possession = row[1]

					else:
						res = rebound_regex.search(row[5])
						if res is not None:
							continue
						else:
							res = foul_regex.search(row[5])
							if res is not None: # foul
								continue
							else:
								res = turnover_regex.search(row[5])
								if res is not None:  # turnover
									if last_writen_possession == row[1]:
										possessions_index -= 1
									elif last_writen_possession == row[1] + ' ':
										possessions_arr[possessions_index] = '0'  # place holder
										possessions_index += 1

									# possessions_arr[possessions_index] += 0
									possessions_index += 1
									last_writen_possession = row[1]
								else:
									res = timeout_regex.search(row[5])
									if res is not None:
										continue
									else:
										res = player_substitution_regex.search(row[5])
										if res is not None:
											continue
				if possessions_arr[0] == '-' and possessions_index == 3:  # who starts the
					possessions_arr[0] = row[1]

			else:  # game comment
				row_values[1:3] = [None]*2
				row_values[7] = '-'  # HorA_ACTION
				row_values[9] = row[2]  # DESCRIPTION
				res = start_of_nth_q_regex.search(row[2])

				if res is not None:  # start of a quarter
					cur_quarter = res.group(1)
					last_writen_possession = last_writen_possession + ' '
					if res.group(3) == 'overtime':
						cur_quarter += 'OT'
					possessions_arr[possessions_index] = cur_quarter
					possessions_index += 1
				else:
					res = jump_ball_regex.search(row[2])
					if res is not None:  # jump ball
						continue

			# insert_arr[row_num] = tuple(row_values)  # "('{0[0]}', {0[1]}, {0[2]}, {0[3]}, {0[4]}, {0[5]}, '{0[6]}', '{0[7]}', {0[8]},
			# '{0[9]}')".format(row_values)
			row_num += 1

		#  print(sum(possessions_arr[0:possessions_index:2]), sum(possessions_arr[1:possessions_index:2]))
		# conn.executemany("INSERT INTO pbp_table VALUES (?,?,?,?,?,?,?,?,?,?)", insert_arr[0:row_num])
		# conn.commit()
		# db_cursor.close()
		# conn.close()
	return possessions_arr[0:possessions_index]


def check_pbp(game_tag, season):
	game_possessions = pbp_to_possessions("Games/{}-{}/{}.csv".format(season-1, season, game_tag))
	a_poss = [i for i in game_possessions[0::2] if type(i) == int]
	b_poss = [i for i in game_possessions[1::2] if type(i) == int]
	print(sum(a_poss), sum(b_poss))


# if __name__ == '__main__':

	# check_pbp('200710310DEN', 2008)

	# start_time = time.time()
	# Playoffs_Parser('https://www.basketball-reference.com')
	# playoffs_series_parser('https://www.basketball-reference.com')
	# games_quarter_scores('https://www.basketball-reference.com')
	# print(time.time() - start_time, time_var)

	# start_time = time.time()
	# Games_Parser()
	# print(time.time() - start_time, time_var)

	# Game_Parser('https://www.basketball-reference.com/boxscores/pbp/201706010GSW.html')
	# print(pbp_to_possessions('Games/201706010GSW.csv'))
	# Game_Parser('https://www.basketball-reference.com/boxscores/pbp/201210310DET.html')
	# start_time = time.time()
	# print(pbp_to_possessions('Games/201210310DET.csv'))
	# print(time.time() - start_time)
	# Games_Parser()

