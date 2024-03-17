from aco import ACO_VRP,ACO_TSP
from acs import ACS_VRP,ACS_TSP
from woa_v2 import WOA_VRP, WOA_VNS_VRP, WOA_TSP, WOA_VNS_TSP
from koneksi import ConDB
import matplotlib.pyplot as plt
import random
import math
import copy
import time
import datetime


def main(tourid, idhotel, dwaktu, drating, dtarif, travel_days, algorithm):
    db = ConDB()
    
    start = time.time()
    hotel = db.HotelbyID(idhotel)
    tur = db.WisatabyID(tourid)
    timematrix = db.TimeMatrixbyID(hotel._id, tourid)
    
    depart_time = datetime.time(8, 0, 0)
    max_travel_time = datetime.time(20, 0, 0)

    aco_algorithms = [
        'ant-colony-system-acs-tsp',
        'ant-colony-optimization-aco-tsp',
        'ant-colony-system-acs-vrp',
        'ant-colony-optimization-aco-vrp',
    ]

    if algorithm == "woa-vrp":
        woa_model = WOA_VRP(tourid, idhotel, dwaktu, dtarif, drating, travel_days)
        solution, fitness = woa_model.construct_solution()
    elif algorithm == "woa-tsp":
        woa_model = WOA_TSP(tourid, idhotel, dwaktu, dtarif, drating, travel_days)
        solution, fitness = woa_model.construct_solution()
    elif algorithm == "woa-vns-vrp":
        woa_model = WOA_VNS_VRP(tourid, idhotel, dwaktu, dtarif, drating, travel_days)
        solution, fitness = woa_model.construct_solution()
    elif algorithm == "woa-vns-tsp":
        woa_model = WOA_VNS_TSP(tourid, idhotel, dwaktu, dtarif, drating, travel_days)
        solution, fitness = woa_model.construct_solution()
    elif algorithm == "ant-colony-system-acs-tsp":
        acs_model = ACS_TSP()
        acs_model.set_model(tur, hotel, timematrix, travel_days, depart_time, max_travel_time, degree_waktu=dwaktu,
                            degree_tarif=dtarif, degree_rating=drating)
        solution, fitness = acs_model.construct_solution()
    elif algorithm == "ant-colony-optimization-aco-tsp":
        aco_model = ACO_TSP()
        aco_model.set_model(tur, hotel, timematrix, travel_days, depart_time, max_travel_time, degree_waktu=dwaktu,
                            degree_tarif=dtarif, degree_rating=drating)
        solution, fitness = aco_model.construct_solution()
    elif algorithm == "ant-colony-system-acs-vrp":
        acs_model = ACS_VRP()
        acs_model.set_model(tur, hotel, timematrix, travel_days, depart_time, max_travel_time, degree_waktu=dwaktu,
                            degree_tarif=dtarif, degree_rating=drating)
        solution, fitness = acs_model.construct_solution()
    elif algorithm == "ant-colony-optimization-aco-vrp":
        aco_model = ACO_VRP()
        aco_model.set_model(tur, hotel, timematrix, travel_days, depart_time, max_travel_time, degree_waktu=dwaktu,
                            degree_tarif=dtarif, degree_rating=drating)
        solution, fitness = aco_model.constuct_solution()
    else:
        solution, fitness = 0, 0 #akan menjadi error ketika mengembalikan hasil optimasi
    end = time.time()

    if solution != 0 and algorithm in aco_algorithms:
        for i in range(len(solution)):
            solution[i]['waktu'] = [str(waktu) for waktu in solution[i]['waktu']]
    
    # print('solution')
    # print('------------------------')
    # print(solution)
    # print('Tipe index : ', type(solution[0]['index'][0]))
    # print('Tipe rating : ', type(solution[0]['rating'][0]))
    # print('Tipe tarif : ', type(solution[0]['tarif'][0]))
    # print('------------------------')

    # print("Time    : ", end - start)

    return solution, fitness
