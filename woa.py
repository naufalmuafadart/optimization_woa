import pandas as pd
import random
from datetime import time, timedelta, date, datetime
import numpy as np
import math
import copy

class WOA():
    HOTEL_ID = 101
    DAYS_COUNT = 3

    DOI_DURATION = 0.3
    DOI_COST = 0.2
    DOI_RATING = 0.5

    DEPART_TIME = time(8, 0, 0) # Waktu untuk berangkat dari hotel
    ARRIVING_TIME = time(21, 0, 0) # Waktu sampai hotel

    START_DATE = date(year=2023, month=10, day=1) # Tanggal mulai perjalanan
    START_TIME = time(8, 0, 0)
    END_TIME = time (21, 0, 0)
    AGENTS_COUNT = 10
    MAX_ITERATIONS = 10
    MAX_ELAPSED = timedelta(seconds=1) # Waktu maksimal untuk perulangan VNS

    def __init__(self, ids, hotel_id, doi_duration, doi_cost, doi_rating, days_count):
        self.PREFERENCE_ID = list(map(int, ids)) # => [1,2,3]
        self.HOTEL_ID = hotel_id
        self.DOI_DURATION = doi_duration
        self.DOI_COST = doi_cost
        self.DOI_RATING = doi_rating
        self.DAYS_COUNT = days_count

    def get_attractions_rating(self, ids):
        ratings = []
        for id in ids:
            df_filter = self.df_places[self.df_places['id'] == (id+1)]
            rating = df_filter.iloc[0]['rating']
            ratings.append(float(rating))
        return ratings
    
    def get_attractions_cost(self, ids):
        ratings = []
        for id in ids:
            df_filter = self.df_places[self.df_places['id'] == (id+1)]
            rating = df_filter.iloc[0]['tarif']
            ratings.append(int(rating))
        return ratings

    def generate_initial_population(self):
        agents = []
        for i in range(self.AGENTS_COUNT):
            agent = []
            for j in range(0, len(self.PREFERENCE_ID)*self.DAYS_COUNT):
                agent.append(random.random() * 10)
            agents.append(agent)
        return agents

    def get_min_rating(self):
        min_series = self.df_places.min(numeric_only=True)
        return min_series['rating']
    
    def get_max_rating(self):
        max_series = self.df_places.max(numeric_only=True)
        return max_series['rating']
    
    def get_min_cost(self):
        min_series = self.df_places.min(numeric_only=True)
        return min_series['tarif']
    
    def get_max_cost(self):
        max_series = self.df_places.max(numeric_only=True)
        return max_series['tarif']

    # Fungsi untuk mendapatkan durasi tempat wisata
    def get_time_spent(self, tour_id):
        return self.df_places.iloc[tour_id-1]['durasi']

    # Fungsi untuk mendapatkan waktu tempuh antara 2 tempat
    def get_travel_time(self, a, b):
        df_filter = self.df_timematrix[(self.df_timematrix['id_a'] == a) & (self.df_timematrix['id_b'] == b)]
        try:
            return df_filter.iloc[0]['durasi']
        except:
            return 0
        else:
            return 0
            
    # fungsi untuk mendapatkan mendapatkan rute pada suatu hari
    def get_route_per_day(self, i, agent_day, assigned_ids, current_time, end_time, route):
        # print('End time : ', end_time)
        agent_length = len(agent_day)

        if i == agent_length:
            return route

        if self.PREFERENCE_ID[agent_day[i]] in assigned_ids:
            return self.get_route_per_day(i+1, agent_day, assigned_ids, current_time, end_time, [])

        tour_id = self.PREFERENCE_ID[agent_day[i]]
        # tour_id = 15
        if i == 0:
            travel_time = self.get_travel_time(tour_id, self.HOTEL_ID)
            time_spent = self.get_time_spent(tour_id)
            time_delta = timedelta(seconds=np.int16(travel_time).item() + np.int16(time_spent).item())
            current_time += time_delta
            if current_time < end_time:
                route.append(tour_id)
                return self.get_route_per_day(i+1, agent_day, assigned_ids, current_time, end_time, route)
            return route

        if i != agent_length-1:
            next_tour_id = self.PREFERENCE_ID[agent_day[i+1]]
            travel_time = self.get_travel_time(tour_id, next_tour_id)
            time_spent = self.get_time_spent(tour_id)
            time_delta = timedelta(seconds=np.int16(travel_time).item() + np.int16(time_spent).item())
            current_time += time_delta
            if current_time < end_time:
                route.append(tour_id)
                return self.get_route_per_day(i+1, agent_day, assigned_ids, current_time, end_time, route)
            return route

        travel_time = self.get_travel_time(tour_id, self.HOTEL_ID)
        time_spent = self.get_time_spent(tour_id)
        time_delta = timedelta(seconds=np.int16(travel_time).item() + np.int16(time_spent).item())
        current_time += time_delta
        if current_time < end_time:
            route.append(tour_id)
            return self.get_route_per_day(i+1, agent_day, assigned_ids, current_time, end_time, route)
        return route

    def get_closing_hour(self, attraction_id):
        df_filter = self.df_jadwal[(self.df_jadwal['id_tempat'] == attraction_id) & (self.df_jadwal['hari'] == 'senin')]
        return df_filter.iloc[0]['jam_tutup']

    def is_able_to_assign(self, route_day, attraction_id):
        if len(route_day) == 0:
            return True

        end_time = datetime(
            self.START_DATE.year,
            self.START_DATE.month,
            self.START_DATE.day,
            self.END_TIME.hour,
            self.END_TIME.minute,
            self.END_TIME.second
            )
        
        current_time = datetime(
            self.START_DATE.year,
            self.START_DATE.month,
            self.START_DATE.day,
            self.START_TIME.hour,
            self.START_TIME.minute,
            self.START_TIME.second
            )

        for i in range(len(route_day)):
            if i == 0:
                travel_time = self.get_travel_time(self.HOTEL_ID, route_day[i])
                time_spent = self.get_time_spent(route_day[i])
                time_delta = timedelta(seconds=np.int16(travel_time).item() + np.int16(time_spent).item())
                current_time += time_delta
            else:
                travel_time = self.get_travel_time(route_day[i-1], route_day[i])
                time_spent = self.get_time_spent(route_day[i])
                time_delta = timedelta(seconds=np.int16(travel_time).item() + np.int16(time_spent).item())
                current_time += time_delta
        
        travel_time = self.get_travel_time(route_day[len(route_day)-1], attraction_id)
        time_spent = self.get_time_spent(route_day[i])
        time_delta = timedelta(seconds=np.int16(travel_time).item() + np.int16(time_spent).item())
        current_time += time_delta

        closing_hour = self.get_closing_hour(attraction_id)
        if closing_hour == '00:00':
            closing_hour = '23:59'
        
        hour = closing_hour[:2]
        minute = closing_hour[3:]

        hour = int(hour)
        minute = int(minute)

        closing_hour = datetime(
            self.START_DATE.year,
            self.START_DATE.month,
            self.START_DATE.day,
            hour,
            minute,
            0
            )

        if (current_time < end_time) & (current_time <= closing_hour):
            return True

        return False

    def get_least_agent_day_index(self, attraction_per_day):
        min = 9999
        min_index = None
        for i in range(len(attraction_per_day)):
            if len(attraction_per_day[i]) < min:
                min = len(attraction_per_day[i])
                min_index = i
        return min_index
    
    # Fungsi untuk mendapatkan rute
    def get_route(self, agent):
        agent_per_day_length = int(len(agent) / self.DAYS_COUNT)
        agent_ = agent[:agent_per_day_length]
        order = np.argsort(agent_)
        # agent_ = [x for _, x in sorted(zip(order, agent_))]
        agent_ = [x for _, x in sorted(zip(order, self.PREFERENCE_ID))]
        attraction_per_day = []

        end_time = datetime(
            self.START_DATE.year,
            self.START_DATE.month,
            self.START_DATE.day,
            self.END_TIME.hour,
            self.END_TIME.minute,
            self.END_TIME.second
            )
        
        current_time = datetime(
            self.START_DATE.year,
            self.START_DATE.month,
            self.START_DATE.day,
            self.START_TIME.hour,
            self.START_TIME.minute,
            self.START_TIME.second
            )

        assigned_ids = []

        # Assign array kosong per hari
        for i in range(self.DAYS_COUNT):
            attraction_per_day.append([])

        iteration_count = 0

        while iteration_count < 5:
            for agent_pointer in range(len(agent_)):
                if agent_[agent_pointer] not in assigned_ids:
                    min_agent_day_index = self.get_least_agent_day_index(attraction_per_day)
                    if self.is_able_to_assign(attraction_per_day[min_agent_day_index], agent_[agent_pointer]):
                        assigned_ids.append(agent_[agent_pointer])
                        attraction_per_day[min_agent_day_index].append(agent_[agent_pointer])
            iteration_count += 1

        if len(assigned_ids) < len(agent_):
            for i in range(len(agent_)):
                if agent_[i] not in assigned_ids:
                    min_agent_day_index = self.get_least_agent_day_index(attraction_per_day)
                    attraction_per_day[min_agent_day_index].append(agent_[i])

        # for i in range(self.DAYS_COUNT):
        #     if index >= agent_per_day_length:
        #         break
        #     current_time = datetime(
        #         self.START_DATE.year,
        #         self.START_DATE.month,
        #         self.START_DATE.day,
        #         self.START_TIME.hour,
        #         self.START_TIME.minute,
        #         self.START_TIME.second
        #         )
        #     route = []
        #     j = 0
        #     while current_time < end_time:
        #         if index >= agent_per_day_length:
        #             break
        #         if j == 0:
        #             travel_time = self.get_travel_time(self.HOTEL_ID, self.PREFERENCE_ID[order[index]])
        #             time_spent = self.get_time_spent(self.PREFERENCE_ID[order[index]])
        #             time_delta = timedelta(seconds=np.int16(travel_time).item() + np.int16(time_spent).item())
        #             current_time += time_delta
        #             if current_time < end_time:
        #                 route.append(self.PREFERENCE_ID[order[index]])
        #         else:
        #             travel_time = self.get_travel_time(self.PREFERENCE_ID[order[index-1]], self.PREFERENCE_ID[order[index]])
        #             time_spent = self.get_time_spent(self.PREFERENCE_ID[order[index]])
        #             time_delta = timedelta(seconds=np.int16(travel_time).item() + np.int16(time_spent).item())
        #             current_time += time_delta
        #             if current_time < end_time:
        #                 route.append(self.PREFERENCE_ID[order[index]])
        #         index = index + 1
        #     attraction_per_day.append(route)
        #     assigned_ids.extend(route)

        # for id in self.PREFERENCE_ID:
        #     if id not in assigned_ids:
        #         attraction_per_day[len(attraction_per_day)-1].append(id)

        return attraction_per_day

    # Fungsi untuk mendapatkan total durasi perjalanan selama N hari
    def get_N_day_duration(self, route):
        duration = 0

        for route_day in route:
            i = 0
            for tour_id in route_day:
                if i == 0:
                    duration += self.get_travel_time(self.HOTEL_ID, tour_id)
                elif i != len(route_day)-1:
                    duration += self.get_travel_time(tour_id, route_day[i+1])
                else:
                    duration += self.get_travel_time(tour_id, self.HOTEL_ID)
                i += 1
        return duration

    def get_min_duration(self):
        durations = []
        for agent in self.agents:
            route = self.get_route(agent)
            durations.append(self.get_N_day_duration(route))
        return min(durations)

    def get_max_duration(self):
        durations = []
        for agent in self.agents:
            route = self.get_route(agent)
            durations.append(self.get_N_day_duration(route))
        return max(durations)

    def separate_route_f(self, agent):
        agent_per_day_length = int(len(agent) / self.DAYS_COUNT)
        agent_ = agent[:agent_per_day_length]
        order = np.argsort(agent_)
        # agent_ = [x for _, x in sorted(zip(order, agent_))]
        route = [x for _, x in sorted(zip(order, self.PREFERENCE_ID))]
        # print('route : ', route)
        # return self.get_route(agent)
        
        # r = []
        # for r_ in route:
        #     r.extend(r_)
        # route = r

        attraction_per_day = []

        end_time = datetime(
            self.START_DATE.year,
            self.START_DATE.month,
            self.START_DATE.day,
            self.END_TIME.hour,
            self.END_TIME.minute,
            self.END_TIME.second
            )
        # print('end time :', end_time)
        
        index = 0
        assigned_ids = []
        for i in range(self.DAYS_COUNT):
            if index >= len(route):
                break
            current_time = datetime(
                self.START_DATE.year,
                self.START_DATE.month,
                self.START_DATE.day,
                self.START_TIME.hour,
                self.START_TIME.minute,
                self.START_TIME.second
                )
            # print('current time :', current_time)
            route_ = []
            j = 0
            while current_time < end_time:
                # print('index : ', index)
                if index >= len(route):
                    break
                if j == 0:
                    travel_time = self.get_travel_time(self.HOTEL_ID, route[index])
                    time_spent = self.get_time_spent(route[index])
                    time_delta = timedelta(seconds=np.int16(travel_time).item() + np.int16(time_spent).item())
                    current_time += time_delta
                    
                    if current_time < end_time:
                        route_.append(route[index])
                else:
                    travel_time = self.get_travel_time(route[index-1], route[index])
                    time_spent = self.get_time_spent(route[index])
                    time_delta = timedelta(seconds=np.int16(travel_time).item() + np.int16(time_spent).item())
                    current_time += time_delta
                    
                    if current_time < end_time:
                        route_.append(route[index])
                index = index + 1
            attraction_per_day.append(route_)
            assigned_ids.extend(route_)
        
        for id in self.PREFERENCE_ID:
            if id not in assigned_ids:
                attraction_per_day[len(attraction_per_day)-1].append(id)

        return attraction_per_day
    
    # fungsi untuk mendapatkan nilai v
    def get_v(self, value, max, min, isGreaterBetter):
        return (abs(value - min) / (max - min)) if isGreaterBetter else (abs(max - value) / (max - min))

    # fungsi untuk mendapatkan nilai fitness
    def fitness_function(self, agent):

        # Mendapatkan urutan tempat wisata perharinya
        # route = self.get_route(agent)
        route = self.separate_route_f(agent)

        # Menghitung total durasi perjalanan selama N hari
        duration = self.get_N_day_duration(route)

        # filter dataset berdasarkan rute
        assigned_ids = []
        for day_route in route:
            assigned_ids.extend(day_route)
        df_temp = self.df_places[self.df_places['id'].isin(assigned_ids)]

        # Menghitung popularitas tempat wisata (rata-rata rating tempat wisata pada rute)
        popularity = df_temp.loc[:, 'rating'].mean()

        # Menghitung tarif tempat wisata (rata-rata biaya tempat wisata pada rute)
        cost = df_temp.loc[:, 'tarif'].mean()

        # Menghitung nilai MAUT
        MAUT = 0
        print(f'Rating : {popularity}')
        print(f'Cost : {cost}')
        MAUT = MAUT + self.get_v(duration, self.MAX_DURATION, self.MIN_DURATION, False) * self.DOI_DURATION
        MAUT = MAUT + self.get_v(popularity, self.MAX_RATING, self.MIN_RATING, True) * self.DOI_RATING
        MAUT = MAUT + self.get_v(cost, self.MAX_COST, self.MIN_COST, False) * self.DOI_COST
        return MAUT

    # Fungsi untuk mendapatkan agen terbaik
    def get_best_agent(self):
        index = -1
        max = 0
        i = 0
        for agent in self.agents:
            fitness_value = self.fitness_function(agent)
            if fitness_value > max:
                max = fitness_value
                index = i
            i += 1
        return (max, self.agents[index])

    def WOA(self, min_x, max_x, agents):
        t = 0

        # Fbest : nilai fitness terbaik
        # Xbest : agen terbaik
        Fbest, Xbest = self.get_best_agent()
        agent_dimension = int(len(Xbest) / self.DAYS_COUNT) # banyak tempat wisata sesuai preferensi user
        fitness_values = []

        # Menyimpan nilai fitness untuk setiap agen
        for i in range(len(agents)):
            fitness_values.append(self.fitness_function(agents[i]))

        # print("Initial best fitness = %.5f" % Fbest)
        while t < self.MAX_ITERATIONS:

            a = 2 * (1 - t / self.MAX_ITERATIONS)
            a2 = -1 + t * ((-1)/self.MAX_ITERATIONS)

            i = 0
            for agent in agents:
                A = 2 * a * random.random() - a
                C = 2 * random.random()
                b = 1
                l = (a2-1)*random.random()+1

                D = [0.0 for k in range(agent_dimension)]
                D1 = [0.0 for k in range(agent_dimension)]
                Xnew = [0.0 for k in range(agent_dimension)]
                Xrand = [0.0 for k in range(agent_dimension)]
                p = random.random()
                if p < 0.5:
                    if abs(a) >= 1: # search for prey
                        p = random.randint(0, self.AGENTS_COUNT-1)
                        while (p==i):
                            p = random.randint(0, self.AGENTS_COUNT-1)

                        Xrand = agents[p]

                        for j in range(agent_dimension):
                            D[j] = abs(C * Xrand[j] - agent[j])
                            Xnew[j] = Xrand[j] - A * D[j]
                    else: # encircling prey
                        for j in range(agent_dimension):
                            D[j] = abs(C * Xrand[j] - agent[j])
                            Xnew[j] = Xrand[j] - A * D[j]
                else: # bubble net attacking
                    for j in range(agent_dimension):
                        D1[j] = abs(Xbest[j] - agent[j])
                        Xnew[j] = D1[j] * math.exp(b * l) * math.cos(2 * math.pi * l) + Xbest[j]

                for j in range(agent_dimension):
                    agent[j] = Xnew[j]
                i += 1

            for i in range(len(agents)):
                # jika Xnew < minx atau Xnew > maxx
                for j in range(agent_dimension):
                    agents[i][j] = max(agents[i][j], min_x)
                    agents[i][j] = min(agents[i][j], max_x)

                fitness_values[i] = self.fitness_function(agents[i])

                if (fitness_values[i] > Fbest):
                    Xbest = copy.copy(agents[i])
                    Fbest = fitness_values[i]

            # print("Iteration = " + str(t) + " | best fitness = %.5f" % Fbest)
            t += 1
        return Xbest, Fbest

    def get_time_line(self, l):
        if len(l) == 0:
            return []
        current_time = datetime(2023, 11, 19, 8, 0, 0)
        time_line = [current_time.strftime('%H:%M:%S')]

        i = 0
        while i < len(l):
            if i == 0:
                travel_time = self.get_travel_time(self.HOTEL_ID, l[i])
                time_delta = timedelta(seconds=np.int16(travel_time).item())
                current_time += time_delta
                time_line.append(current_time.strftime('%H:%M:%S'))
            elif i != len(l)-1:
                time_spent = self.get_time_spent(l[i])
                travel_time = self.get_travel_time(l[i], l[i+1])
                time_delta = timedelta(seconds=np.int16(travel_time).item() + np.int16(time_spent).item())
                current_time += time_delta
                time_line.append(current_time.strftime('%H:%M:%S'))
            else:
                time_spent = self.get_time_spent(l[i])
                time_delta = timedelta(seconds=np.int16(time_spent).item())
                current_time += time_delta
                time_line.append(current_time.strftime('%H:%M:%S'))
                
                travel_time = self.get_travel_time(l[i], self.HOTEL_ID)
                time_delta = timedelta(seconds=np.int16(travel_time).item())
                current_time += time_delta
                time_line.append(current_time.strftime('%H:%M:%S'))
            i += 1

        return time_line

    def construct_solution(self):
        #generate initial population
        self.agents = self.generate_initial_population()

        # get dataset
        self.df_places = pd.read_csv('./Dataset/places.csv')
        self.df_timematrix = pd.read_csv('./Dataset/place_timematrix.csv')
        self.df_jadwal = pd.read_csv('./Dataset/place_jadwal.csv')

        # get max min atrribute
        self.MIN_RATING = self.get_min_rating()
        self.MAX_RATING = self.get_max_rating()
        self.MIN_COST = self.get_min_cost()
        self.MAX_COST = self.get_max_cost()
        self.MIN_DURATION = self.get_min_duration()
        self.MAX_DURATION = self.get_max_duration()

        agents_ = copy.deepcopy(self.agents)
        Xbest, Fbest = self.WOA(0.0, 10.0, agents_)
        route = self.get_route(Xbest)
        fitness_value = self.fitness_function(Xbest)
        duration = self.get_N_day_duration(route)
        print(f'Min rating : {self.MIN_RATING}')
        print(f'Max rating : {self.MAX_RATING}')
        print(f'Min cost : {self.MIN_COST}')
        print(f'Max cost : {self.MAX_COST}')
        print(f'Min duration : {self.MIN_DURATION}')
        print(f'Max duration : {self.MAX_DURATION}')
        print(f'Fitness value : {fitness_value}')
        print(f'Duration : {duration}')
        # print('Route : ', route)

        output = {}
        output['results'] = []
        for day_route in route:
            output['results'].append({
                'index': day_route,
                'waktu': self.get_time_line(day_route),
                'rating': self.get_attractions_rating(day_route),
                'tarif': self.get_attractions_cost(day_route),
            })
        
        return output['results'], Fbest

class WOA_VRP(WOA):
    pass

class WOA_VNS_VRP(WOA):
    MAX_ELAPSED= timedelta(milliseconds=500)
    
    def n1(self, l, min_index = None, max_index = None):
        if not isinstance(l, list): # Parameter l hanya boleh diisi list
            raise ValueError('Function n1 only receive list')

        if not all(isinstance(n, float) for n in l): # Semua nilai pada list l harus bertipe float
            raise ValueError('Function n1 only receive float list')

        if len(l) < 2: # List l minimal mempunyai 2 elemen
            raise ValueError('Function n1 only receive list with at least 2 elements')

        if len(l) == 2: # Jika list hanya terdiri dari 2 elemen
            l.reverse()
            return l

        # Menentukan min index dan max index
        sorted_index = np.argsort(l)
        if min_index is None and max_index is None:
            min_index, max_index = random.sample(list(sorted_index), k=2)

        # Menukar min index dan max index jika nilai min index lebih dari max index
        if min_index > max_index:
            min_index, max_index = max_index, min_index

        reversed_l = l[min_index:max_index+1] # Mendapatkan nilai list dari min index sampai max index
        reversed_l.reverse() # Membalik urutan list
        l[min_index:max_index+1] = reversed_l # Assign list yang sudah dibalik

        return l

    def n2(self, l):
        if not isinstance(l, list): # Parameter l hanya boleh diisi list
            raise ValueError('Function n2 only receive list')

        if not all(isinstance(n, float) for n in l): # Semua nilai pada list l harus bertipe float
            raise ValueError('Function n2 only receive float list')

        if len(l) < 3: # List l minimal mempunyai 3 elemen
            raise ValueError('Function n2 only receive list with at least 3 elements')

        sorted_index = np.argsort(l)
        indexs = random.sample(list(sorted_index), k=3) # Mendapatkan 3 index acak
        indexs.sort()

        s1 = self.n1(l, indexs[0], indexs[1]) # Menjalankan n1 untuk list l pada indexs[0] dan indexs[1]
        s2 = self.n1(s1, indexs[0], indexs[2]) # Menjalankan n1 untuk list s1 pada indexs[0] dan indexs[2]
        s_ = self.n1(s2, indexs[1], indexs[2]) # Menjalankan n1 untuk list s2 pada indexs[1] dan indexs[2]

        return s_

    def n3(self, l):
        if not isinstance(l, list): # Parameter l hanya boleh diisi list
            raise ValueError('Function n3 only receive list')

        if not all(isinstance(n, float) for n in l): # Semua nilai pada list l harus bertipe float
            raise ValueError('Function n3 only receive float list')

        if len(l) < 2: # List l minimal mempunyai 2 elemen
            raise ValueError('Function n3 only receive list with at least 2 elements')

        sorted_index = np.argsort(l)
        min, max = random.sample(list(sorted_index), k=2) # Mendapatkan 2 index secara acak
        l[min], l[max] = l[max], l[min] # Nilai di kedua index ditukar
        return l

    def vns(self,agent):
        start_timestamp = datetime.now()
        agent_ = None
        elapsed = datetime.now() - start_timestamp
        while elapsed < self.MAX_ELAPSED: # sampai maksimal waktu 1 detik
            k = 1
            while k <= 3:
                # Menjalankan n1, n2, atau n3
                if k == 1:
                    agent_ = self.n1(agent)
                elif k == 2:
                    agent_ = self.n2(agent)
                else:
                    agent_ = self.n3(agent)

                fitness_value_agent = self.fitness_function(agent)

                fitness_value_agent_ = self.fitness_function(agent_)

                # jika agent_ lebih baik dari agen saat ini
                if fitness_value_agent_ > fitness_value_agent:
                    agent = agent_
                    k = 1
                else: # jika agent_ tidak lebih baik dari agen saat ini
                    k += 1
            elapsed = datetime.now() - start_timestamp
        return agent_

    def WOA(self, min_x, max_x, agents):
        # print('WOA-VNS')
        t = 0

        # Fbest : nilai fitness terbaik
        # Xbest : agen terbaik
        Fbest, Xbest = self.get_best_agent()
        agent_dimension = int(len(Xbest) / self.DAYS_COUNT) # banyak tempat wisata sesuai preferensi user
        fitness_values = []

        # Menyimpan nilai fitness untuk setiap agen
        for i in range(len(agents)):
            fitness_values.append(self.fitness_function(agents[i]))

        # print("Initial best fitness = %.5f" % Fbest)
        while t < self.MAX_ITERATIONS:

            a = 2 * (1 - t / self.MAX_ITERATIONS)
            a2 = -1 + t * ((-1)/self.MAX_ITERATIONS)

            i = 0
            for agent in agents:
                A = 2 * a * random.random() - a
                C = 2 * random.random()
                b = 1
                l = (a2-1)*random.random()+1

                D = [0.0 for k in range(agent_dimension)]
                D1 = [0.0 for k in range(agent_dimension)]
                Xnew = [0.0 for k in range(agent_dimension)]
                Xrand = [0.0 for k in range(agent_dimension)]
                p = random.random()
                r = random.random()
                if p < 0.5:
                    if abs(a) >= 1: # search for prey
                        p = random.randint(0, self.AGENTS_COUNT-1)
                        while (p==i):
                            p = random.randint(0, self.AGENTS_COUNT-1)

                        Xrand = agents[p]

                        for j in range(agent_dimension):
                            D[j] = abs(C * Xrand[j] - agent[j])
                            Xnew[j] = Xrand[j] - A * D[j]
                    else: # encircling prey
                        if r < 0.5:
                            for j in range(agent_dimension):
                                D[j] = abs(C * Xrand[j] - agent[j])
                                Xnew[j] = Xrand[j] - A * D[j]
                        else:
                            Xnew = self.vns(agent)
                else: # bubble net attacking
                    if r < 0.5:
                        for j in range(agent_dimension):
                            D1[j] = abs(Xbest[j] - agent[j])
                            Xnew[j] = D1[j] * math.exp(b * l) * math.cos(2 * math.pi * l) + Xbest[j]
                    else:
                        Xnew = self.vns(agent)

                for j in range(agent_dimension):
                    agent[j] = Xnew[j]
                i += 1

            for i in range(len(agents)):
                # jika Xnew < minx atau Xnew > maxx
                for j in range(agent_dimension):
                    agents[i][j] = max(agents[i][j], min_x)
                    agents[i][j] = min(agents[i][j], max_x)

                fitness_values[i] = self.fitness_function(agents[i])

                if (fitness_values[i] > Fbest):
                    Xbest = copy.copy(agents[i])
                    Fbest = fitness_values[i]

            # print("Iteration = " + str(t) + " | best fitness = %.5f" % Fbest)
            t += 1
        return Xbest, Fbest

class WOA_TSP(WOA):
    def get_route(self, agent):
        agent_per_day_length = int(len(agent) / self.DAYS_COUNT)
        agent_ = agent[:agent_per_day_length]
        order = np.argsort(agent_)
        agent_ = [x for _, x in sorted(zip(order, self.PREFERENCE_ID))]
        return [agent_]

    def fitness_function(self, agent):
        # Mendapatkan urutan tempat wisata perharinya
        route = self.get_route(agent)

        # Menghitung total durasi perjalanan selama N hari
        duration = self.get_N_day_duration(route)
        return duration

    # Fungsi untuk mendapatkan agen terbaik
    def get_best_agent(self):
        index = -1
        min = 99999
        i = 0
        for agent in self.agents:
            fitness_value = self.fitness_function(agent)
            if fitness_value < min:
                min = fitness_value
                index = i
            i += 1
        return (min, self.agents[index])

    def separate_route(self, route):
        
        r = []
        for r_ in route:
            r.extend(r_)
        route = r

        attraction_per_day = []

        end_time = datetime(
            self.START_DATE.year,
            self.START_DATE.month,
            self.START_DATE.day,
            self.END_TIME.hour,
            self.END_TIME.minute,
            self.END_TIME.second
            )
        # print('end time :', end_time)
        
        index = 0
        assigned_ids = []
        for i in range(self.DAYS_COUNT):
            if index >= len(route):
                break
            current_time = datetime(
                self.START_DATE.year,
                self.START_DATE.month,
                self.START_DATE.day,
                self.START_TIME.hour,
                self.START_TIME.minute,
                self.START_TIME.second
                )
            # print('current time :', current_time)
            route_ = []
            j = 0
            while current_time < end_time:
                # print('index : ', index)
                if index >= len(route):
                    break
                if j == 0:
                    travel_time = self.get_travel_time(self.HOTEL_ID, route[index])
                    time_spent = self.get_time_spent(route[index])
                    time_delta = timedelta(seconds=np.int16(travel_time).item() + np.int16(time_spent).item())
                    current_time += time_delta
                    # print('current time :', current_time)
                    if current_time < end_time:
                        route_.append(route[index])
                else:
                    travel_time = self.get_travel_time(route[index-1], route[index])
                    time_spent = self.get_time_spent(route[index])
                    time_delta = timedelta(seconds=np.int16(travel_time).item() + np.int16(time_spent).item())
                    current_time += time_delta
                    # print('current time :', current_time)
                    if current_time < end_time:
                        route_.append(route[index])
                index = index + 1
            attraction_per_day.append(route_)
            assigned_ids.extend(route_)
        # attraction_per_day[len(attraction_per_day)-1].append(self.PREFERENCE_ID[order[len(order)-1]])
        for id in self.PREFERENCE_ID:
            if id not in assigned_ids:
                attraction_per_day[len(attraction_per_day)-1].append(id)

        return attraction_per_day
    
    def construct_solution(self):
        #generate initial population
        self.agents = self.generate_initial_population()

        # get dataset
        self.df_places = pd.read_csv('./Dataset/places.csv')
        self.df_timematrix = pd.read_csv('./Dataset/place_timematrix.csv')

        # get max min atrribute
        self.MIN_RATING = self.get_min_rating()
        self.MAX_RATING = self.get_max_rating()
        self.MIN_COST = self.get_min_cost()
        self.MAX_COST = self.get_max_cost()
        self.MIN_DURATION = self.get_min_duration()
        self.MAX_DURATION = self.get_max_duration()

        agents_ = copy.deepcopy(self.agents)
        Xbest, Fbest = self.WOA(0.0, 10.0, agents_)
        route = self.get_route(Xbest)
        route = self.separate_route(route)
        # print('Route : ', route)
        fitness_value = self.fitness_function(Xbest)
        duration = self.get_N_day_duration(route)
        print(f'Min rating : {self.MIN_RATING}')
        print(f'Max rating : {self.MAX_RATING}')
        print(f'Min cost : {self.MIN_COST}')
        print(f'Max cost : {self.MAX_COST}')
        print(f'Min duration : {self.MIN_DURATION}')
        print(f'Max duration : {self.MAX_DURATION}')
        print(f'Fitness value : {fitness_value}')
        print(f'Duration : {duration}')

        output = {}
        output['results'] = []
        for day_route in route:
            output['results'].append({
                'index': day_route,
                'waktu': self.get_time_line(day_route),
                'rating': self.get_attractions_rating(day_route),
                'tarif': self.get_attractions_cost(day_route),
            })
        
        return output['results'], Fbest
    
    def WOA(self, min_x, max_x, agents):
        t = 0

        # Fbest : nilai fitness terbaik
        # Xbest : agen terbaik
        Fbest, Xbest = self.get_best_agent()
        agent_dimension = int(len(Xbest) / self.DAYS_COUNT) # banyak tempat wisata sesuai preferensi user
        fitness_values = []

        # Menyimpan nilai fitness untuk setiap agen
        for i in range(len(agents)):
            fitness_values.append(self.fitness_function(agents[i]))

        # print("Initial best fitness = %.5f" % Fbest)
        while t < self.MAX_ITERATIONS:

            a = 2 * (1 - t / self.MAX_ITERATIONS)
            a2 = -1 + t * ((-1)/self.MAX_ITERATIONS)

            i = 0
            for agent in agents:
                A = 2 * a * random.random() - a
                C = 2 * random.random()
                b = 1
                l = (a2-1)*random.random()+1

                D = [0.0 for k in range(agent_dimension)]
                D1 = [0.0 for k in range(agent_dimension)]
                Xnew = [0.0 for k in range(agent_dimension)]
                Xrand = [0.0 for k in range(agent_dimension)]
                p = random.random()
                if p < 0.5:
                    if abs(a) >= 1: # search for prey
                        p = random.randint(0, self.AGENTS_COUNT-1)
                        while (p==i):
                            p = random.randint(0, self.AGENTS_COUNT-1)

                        Xrand = agents[p]

                        for j in range(agent_dimension):
                            D[j] = abs(C * Xrand[j] - agent[j])
                            Xnew[j] = Xrand[j] - A * D[j]
                    else: # encircling prey
                        for j in range(agent_dimension):
                            D[j] = abs(C * Xrand[j] - agent[j])
                            Xnew[j] = Xrand[j] - A * D[j]
                else: # bubble net attacking
                    for j in range(agent_dimension):
                        D1[j] = abs(Xbest[j] - agent[j])
                        Xnew[j] = D1[j] * math.exp(b * l) * math.cos(2 * math.pi * l) + Xbest[j]

                for j in range(agent_dimension):
                    agent[j] = Xnew[j]
                i += 1

            for i in range(len(agents)):
                # jika Xnew < minx atau Xnew > maxx
                for j in range(agent_dimension):
                    agents[i][j] = max(agents[i][j], min_x)
                    agents[i][j] = min(agents[i][j], max_x)

                fitness_values[i] = self.fitness_function(agents[i])

                if (fitness_values[i] < Fbest):
                    Xbest = copy.copy(agents[i])
                    Fbest = fitness_values[i]

            # print("Iteration = " + str(t) + " | best fitness = %.5f" % Fbest)
            t += 1
        return Xbest, Fbest

class WOA_VNS_TSP(WOA):
    MAX_ELAPSED= timedelta(milliseconds=500)

    def get_route(self, agent):
        agent_per_day_length = int(len(agent) / self.DAYS_COUNT)
        agent_ = agent[:agent_per_day_length]
        order = np.argsort(agent_)
        agent_ = [x for _, x in sorted(zip(order, self.PREFERENCE_ID))]
        return [agent_]

    def fitness_function(self, agent):
        # Mendapatkan urutan tempat wisata perharinya
        route = self.get_route(agent)

        # Menghitung total durasi perjalanan selama N hari
        duration = self.get_N_day_duration(route)
        return duration

    # Fungsi untuk mendapatkan agen terbaik
    def get_best_agent(self):
        index = -1
        min = 99999
        i = 0
        for agent in self.agents:
            fitness_value = self.fitness_function(agent)
            if fitness_value < min:
                min = fitness_value
                index = i
            i += 1
        return (min, self.agents[index])
    
    def n1(self, l, min_index = None, max_index = None):
        if not isinstance(l, list): # Parameter l hanya boleh diisi list
            raise ValueError('Function n1 only receive list')

        if not all(isinstance(n, float) for n in l): # Semua nilai pada list l harus bertipe float
            raise ValueError('Function n1 only receive float list')

        if len(l) < 2: # List l minimal mempunyai 2 elemen
            raise ValueError('Function n1 only receive list with at least 2 elements')

        if len(l) == 2: # Jika list hanya terdiri dari 2 elemen
            l.reverse()
            return l

        # Menentukan min index dan max index
        sorted_index = np.argsort(l)
        if min_index is None and max_index is None:
            min_index, max_index = random.sample(list(sorted_index), k=2)

        # Menukar min index dan max index jika nilai min index lebih dari max index
        if min_index > max_index:
            min_index, max_index = max_index, min_index

        reversed_l = l[min_index:max_index+1] # Mendapatkan nilai list dari min index sampai max index
        reversed_l.reverse() # Membalik urutan list
        l[min_index:max_index+1] = reversed_l # Assign list yang sudah dibalik

        return l

    def n2(self, l):
        if not isinstance(l, list): # Parameter l hanya boleh diisi list
            raise ValueError('Function n2 only receive list')

        if not all(isinstance(n, float) for n in l): # Semua nilai pada list l harus bertipe float
            raise ValueError('Function n2 only receive float list')

        if len(l) < 3: # List l minimal mempunyai 3 elemen
            raise ValueError('Function n2 only receive list with at least 3 elements')

        sorted_index = np.argsort(l)
        indexs = random.sample(list(sorted_index), k=3) # Mendapatkan 3 index acak
        indexs.sort()

        s1 = self.n1(l, indexs[0], indexs[1]) # Menjalankan n1 untuk list l pada indexs[0] dan indexs[1]
        s2 = self.n1(s1, indexs[0], indexs[2]) # Menjalankan n1 untuk list s1 pada indexs[0] dan indexs[2]
        s_ = self.n1(s2, indexs[1], indexs[2]) # Menjalankan n1 untuk list s2 pada indexs[1] dan indexs[2]

        return s_

    def n3(self, l):
        if not isinstance(l, list): # Parameter l hanya boleh diisi list
            raise ValueError('Function n3 only receive list')

        if not all(isinstance(n, float) for n in l): # Semua nilai pada list l harus bertipe float
            raise ValueError('Function n3 only receive float list')

        if len(l) < 2: # List l minimal mempunyai 2 elemen
            raise ValueError('Function n3 only receive list with at least 2 elements')

        sorted_index = np.argsort(l)
        min, max = random.sample(list(sorted_index), k=2) # Mendapatkan 2 index secara acak
        l[min], l[max] = l[max], l[min] # Nilai di kedua index ditukar
        return l

    def vns(self,agent):
        start_timestamp = datetime.now()
        agent_ = None
        elapsed = datetime.now() - start_timestamp
        while elapsed < self.MAX_ELAPSED: # sampai maksimal waktu 1 detik
            k = 1
            while k <= 3:
                # Menjalankan n1, n2, atau n3
                if k == 1:
                    agent_ = self.n1(agent)
                elif k == 2:
                    agent_ = self.n2(agent)
                else:
                    agent_ = self.n3(agent)

                fitness_value_agent = self.fitness_function(agent)

                fitness_value_agent_ = self.fitness_function(agent_)

                # jika agent_ lebih baik dari agen saat ini
                if fitness_value_agent_ > fitness_value_agent:
                    agent = agent_
                    k = 1
                else: # jika agent_ tidak lebih baik dari agen saat ini
                    k += 1
            elapsed = datetime.now() - start_timestamp
        return agent_

    def WOA(self, min_x, max_x, agents):
        # print('WOA-VNS')
        t = 0

        # Fbest : nilai fitness terbaik
        # Xbest : agen terbaik
        Fbest, Xbest = self.get_best_agent()
        agent_dimension = int(len(Xbest) / self.DAYS_COUNT) # banyak tempat wisata sesuai preferensi user
        fitness_values = []

        # Menyimpan nilai fitness untuk setiap agen
        for i in range(len(agents)):
            fitness_values.append(self.fitness_function(agents[i]))

        # print("Initial best fitness = %.5f" % Fbest)
        while t < self.MAX_ITERATIONS:

            a = 2 * (1 - t / self.MAX_ITERATIONS)
            a2 = -1 + t * ((-1)/self.MAX_ITERATIONS)

            i = 0
            for agent in agents:
                A = 2 * a * random.random() - a
                C = 2 * random.random()
                b = 1
                l = (a2-1)*random.random()+1

                D = [0.0 for k in range(agent_dimension)]
                D1 = [0.0 for k in range(agent_dimension)]
                Xnew = [0.0 for k in range(agent_dimension)]
                Xrand = [0.0 for k in range(agent_dimension)]
                p = random.random()
                r = random.random()
                if p < 0.5:
                    if abs(a) >= 1: # search for prey
                        p = random.randint(0, self.AGENTS_COUNT-1)
                        while (p==i):
                            p = random.randint(0, self.AGENTS_COUNT-1)

                        Xrand = agents[p]

                        for j in range(agent_dimension):
                            D[j] = abs(C * Xrand[j] - agent[j])
                            Xnew[j] = Xrand[j] - A * D[j]
                    else: # encircling prey
                        if r < 0.5:
                            for j in range(agent_dimension):
                                D[j] = abs(C * Xrand[j] - agent[j])
                                Xnew[j] = Xrand[j] - A * D[j]
                        else:
                            Xnew = self.vns(agent)
                else: # bubble net attacking
                    if r < 0.5:
                        for j in range(agent_dimension):
                            D1[j] = abs(Xbest[j] - agent[j])
                            Xnew[j] = D1[j] * math.exp(b * l) * math.cos(2 * math.pi * l) + Xbest[j]
                    else:
                        Xnew = self.vns(agent)

                for j in range(agent_dimension):
                    agent[j] = Xnew[j]
                i += 1

            for i in range(len(agents)):
                # jika Xnew < minx atau Xnew > maxx
                for j in range(agent_dimension):
                    agents[i][j] = max(agents[i][j], min_x)
                    agents[i][j] = min(agents[i][j], max_x)

                fitness_values[i] = self.fitness_function(agents[i])

                if (fitness_values[i] < Fbest):
                    Xbest = copy.copy(agents[i])
                    Fbest = fitness_values[i]

            # print("Iteration = " + str(t) + " | best fitness = %.5f" % Fbest)
            t += 1
        return Xbest, Fbest

    def separate_route(self, route):
        
        r = []
        for r_ in route:
            r.extend(r_)
        route = r

        attraction_per_day = []

        end_time = datetime(
            self.START_DATE.year,
            self.START_DATE.month,
            self.START_DATE.day,
            self.END_TIME.hour,
            self.END_TIME.minute,
            self.END_TIME.second
            )
        # print('end time :', end_time)
        
        index = 0
        assigned_ids = []
        for i in range(self.DAYS_COUNT):
            if index >= len(route):
                break
            current_time = datetime(
                self.START_DATE.year,
                self.START_DATE.month,
                self.START_DATE.day,
                self.START_TIME.hour,
                self.START_TIME.minute,
                self.START_TIME.second
                )
            # print('current time :', current_time)
            route_ = []
            j = 0
            while current_time < end_time:
                # print('index : ', index)
                if index >= len(route):
                    break
                if j == 0:
                    travel_time = self.get_travel_time(self.HOTEL_ID, route[index])
                    time_spent = self.get_time_spent(route[index])
                    time_delta = timedelta(seconds=np.int16(travel_time).item() + np.int16(time_spent).item())
                    current_time += time_delta
                    print('current time :', current_time)
                    if current_time < end_time:
                        route_.append(route[index])
                else:
                    travel_time = self.get_travel_time(route[index-1], route[index])
                    time_spent = self.get_time_spent(route[index])
                    time_delta = timedelta(seconds=np.int16(travel_time).item() + np.int16(time_spent).item())
                    current_time += time_delta
                    # print('current time :', current_time)
                    if current_time < end_time:
                        route_.append(route[index])
                index = index + 1
            attraction_per_day.append(route_)
            assigned_ids.extend(route_)
        # attraction_per_day[len(attraction_per_day)-1].append(self.PREFERENCE_ID[order[len(order)-1]])
        for id in self.PREFERENCE_ID:
            if id not in assigned_ids:
                attraction_per_day[len(attraction_per_day)-1].append(id)

        return attraction_per_day
    
    def construct_solution(self):
        #generate initial population
        self.agents = self.generate_initial_population()

        # get dataset
        self.df_places = pd.read_csv('./Dataset/places.csv')
        self.df_timematrix = pd.read_csv('./Dataset/place_timematrix.csv')

        # get max min atrribute
        self.MIN_RATING = self.get_min_rating()
        self.MAX_RATING = self.get_max_rating()
        self.MIN_COST = self.get_min_cost()
        self.MAX_COST = self.get_max_cost()
        self.MIN_DURATION = self.get_min_duration()
        self.MAX_DURATION = self.get_max_duration()

        agents_ = copy.deepcopy(self.agents)
        Xbest, Fbest = self.WOA(0.0, 10.0, agents_)
        route = self.get_route(Xbest)
        route = self.separate_route(route)
        fitness_value = self.fitness_function(Xbest)
        duration = self.get_N_day_duration(route)
        print(f'Min rating : {self.MIN_RATING}')
        print(f'Max rating : {self.MAX_RATING}')
        print(f'Min cost : {self.MIN_COST}')
        print(f'Max cost : {self.MAX_COST}')
        print(f'Min duration : {self.MIN_DURATION}')
        print(f'Max duration : {self.MAX_DURATION}')
        print(f'Fitness value : {fitness_value}')
        print(f'Duration : {duration}')

        output = {}
        output['results'] = []
        for day_route in route:
            output['results'].append({
                'index': day_route,
                'waktu': self.get_time_line(day_route),
                'rating': self.get_attractions_rating(day_route),
                'tarif': self.get_attractions_cost(day_route),
            })
        
        return output['results'], Fbest
