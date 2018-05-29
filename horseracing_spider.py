import scrapy
import json
import re
import csv

class horseracingSpider(scrapy.Spider):
	name = 'hkjc'
	allowed_domains = ['http://racing.hkjc.com']

	def start_requests(self):
		headers = {
			'accept-encoding': 'gzip, deflate, sdch, br',
			'accept-language': 'en-US,en;q=0.8,zh-CN;q=0.6,zh;q=0.4',
			'upgrade-insecure-requests': '1',
			'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
			'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
			'cache-control': 'max-age=0',
		}

		with open('url.json', 'r') as file:
			web = json.load(file)
		base = "http://racing.hkjc.com/racing/info/meeting/Results/english/"
		header = ['subject', 'code', 'horsenum', 'win', 'day', 'distance', 'HW', 'Odds', 'Runs', \
				   'JRecent', 'day', "rc", 'RClass', 'NonGPrace', 'Draw', 'RunningPosition1', \
				   'RunningPosition2', 'RunningPosition3', 'RunningPosition4', 'FinishTime']
		with open("horse_data.csv", "w+") as fp:
			wr = csv.writer(fp, dialect='excel')
			wr.writerow(header)
		for link in web:
				url = base + str(link)
				yield scrapy.Request(url=url, headers=headers, callback=self.parse)

	def find_date_from_url(self, urllink):
		try:
			start = urllink.index('english/Local/') + len('english/Local/')
			end = int(start) + 8
			return urllink[start:end]
		except ValueError:
			return ""

	def parse(self, response):
		def convert_to_float_time(frac_str):
			if frac_str.isdigit():
				return float(frac_str)
			else:
				min, sec, subsec = frac_str.split('.')
				whole = float(min) * 60 + float(sec) + float(subsec) / 100
				return whole
		def convert_to_float_length_behind(frac_str):
			if frac_str.isdigit():
				return float(frac_str)
			else:
				try:
					if not frac_str.replace('-', '').replace('/', '').isdigit():
						return 0
					else:
						if frac_str.find('-') == -1:
							num, denom = frac_str.split('/')
							whole = (float(num) / float(denom))
							return whole
						else:
							int, frac = frac_str.split('-')
							num, denom = frac.split('/')
							whole = float(int) + (float(num) / float(denom))
							return whole
				except Exception as e:
					return(e)
		table_rows = {}
		year = str(response.request.url).split("english/Local/")[1][:4]
		region = response.css('td[style="text-align:right;"].racingTitle::text').extract_first()
		field = response.css('div.clearDivFloat.paddingTop5 table.tableBorder0.font13 tr td::text').extract()
		if "sha tin" or "shatin" in region.lower():
			for x in field:
				if "turf" in x.lower():
					rc = 0
				else:
					rc = 1
		elif 'happy valley' in region.lower():
			rc = 2

		distance = response.css('td.divWidth span.number14::text').extract_first()
		ClassOrGroup = response.css('td.divWidth::text').extract_first()
		if 'class' in str(ClassOrGroup).lower():
			RClass = re.findall('\d+', str(ClassOrGroup))[0]
			NonGPrace = str(1)
		elif 'group' in str(ClassOrGroup).lower():
			RClass = str(1)
			NonGPrace = str(0)
		table = response.css('table.tableBorder.trBgBlue.tdAlignC.number12.draggable tr')

		for row in table:
				row_list = row.css('td *::text').extract()
				if len(row_list) >= 9:
					try:
						int(row_list[1])
						yesorno = True
					except ValueError:
						yesorno = False
					if yesorno:
						status  = row_list[0]
						if status.isdigit():
							s = str(response.css('div.boldFont14.color_white.trBgBlue::text').extract_first())
							racenum = s[s.find("(") + 1:s.find(")")]
							table_rows['subject'] = str(year) + str("{0:0=3d}".format(int(racenum)))
							table_rows['code'] = str(year) + str("{0:0=3d}".format(int(racenum))) + str(row_list[1]).zfill(2)
							length = len(row_list)
							table_rows['horsenum'] = int(row_list[1])
							if row_list[0] == '1':
								table_rows['win'] = 1
							else:
								table_rows['win'] = 2
							table_rows['day'] = ''
							table_rows['distance'] = distance.split('M')[0]
							table_rows['JW'] = str(row_list[6])
							table_rows['HW'] = str(row_list[7])
							table_rows['Odds'] = str(row_list[length-1])
							table_rows['Runs'] = ''
							table_rows['JRecent'] = convert_to_float_length_behind(row_list[9])
							table_rows['day'] = str(response.request.url).split("english/Local/")[1][:8]
							table_rows["rc"] = rc
							table_rows['RClass'] = RClass
							table_rows['NonGPrace'] = NonGPrace
							table_rows['Draw'] = str(row_list[8])
							if length >= 15:
								table_rows['RunningPosition1'] = str(row_list[10])
								table_rows['RunningPosition2'] = str(row_list[11])
								table_rows['RunningPosition3'] = str(row_list[12])
								table_rows['RunningPosition4'] = str(row_list[13])
							elif length == 14:
								table_rows['RunningPosition1'] = str(row_list[10])
								table_rows['RunningPosition2'] = str(row_list[11])
								table_rows['RunningPosition3'] = str(row_list[12])
								table_rows['RunningPosition4'] = ''
							elif length == 13:
								table_rows['RunningPosition1'] = str(row_list[10])
								table_rows['RunningPosition2'] = str(row_list[11])
								table_rows['RunningPosition3'] = ''
								table_rows['RunningPosition4'] = ''
							elif length == 12:
								table_rows['RunningPosition1'] = str(row_list[10])
								table_rows['RunningPosition2'] = ''
								table_rows['RunningPosition3'] = ''
								table_rows['RunningPosition4'] = ''
							elif length == 11:
								table_rows['RunningPosition1'] = ''
								table_rows['RunningPosition2'] = ''
								table_rows['RunningPosition3'] = ''
								table_rows['RunningPosition4'] = ''
							try:
								table_rows['FinishTime'] = str(convert_to_float_time(row_list[length-2]))
							except ValueError:
								pass
							myCsvRow = [table_rows['subject'],table_rows['code'],table_rows['horsenum'],\
										table_rows['win'],table_rows['day'],table_rows['distance'],\
										table_rows['HW'],table_rows['Odds'],\
										table_rows['Runs'],table_rows['JRecent'],table_rows['day'],\
										table_rows["rc"],table_rows['RClass'],table_rows['NonGPrace'],\
										table_rows['Draw'],table_rows['RunningPosition1'],\
										table_rows['RunningPosition2'],table_rows['RunningPosition3'],\
										table_rows['RunningPosition4'],table_rows['FinishTime']]
							with open("horse_data.csv", "a+") as fp:
								wr = csv.writer(fp, dialect='excel')
								wr.writerow(myCsvRow)
							myCsvRow = None
							table_rows = {}
						else:
							pass
				else:
					pass
