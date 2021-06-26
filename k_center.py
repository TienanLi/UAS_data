import pickle
import random
import matplotlib
from side_function import covert_coordinate_from_4326_to_MA,maintenance_depots,cal_cost,network_algorithm,draw_figures,\
    read_raster, raster_coverage

font = {'size'   : 24}
matplotlib.rc('font', **font)

def read_point_and_process():
    point_set = pickle.load(open('point_set.obj', 'rb'))
    start_time=[p[1] for p in point_set]
    duration=[p[2] for p in point_set]
    severity=[p[3] for p in point_set]
    sub_type=[p[4] for p in point_set]
    point_set=[p[0] for p in point_set]
    # print('incidents:',len(point_set))
    new_point_set=[(p[0]+random.uniform(-1e-09,1e-09), p[1]+random.uniform(-1e-09, 1e-09)) for p in point_set]
    global TH,r
    TH = 0.05 # how much point can be ignored, i.e., 0.05 means >=95% coverage
    for r in [5, 10]:
        print('radius:',r)
        r = r * 1000
        r = r * 1.60934
        greedy_find(new_point_set, start_time, duration, severity, sub_type)
    # greedy_generate(point_set)

def greedy_find(point_set,start_time_set,duration_set,severity,sub_type):
    # raster_area1=read_raster('ForLowell/2%floodfinal.tif')
    # raster_area2=read_raster('ForLowell/bob9151mph1.tif')
    # raster_area3=read_raster('ForLowell/fld9951mph1.tif')
    m_d_set=maintenance_depots('Maintenance_Depots.shp')

    best_result=[1e8 for i in range(3)]
    for i in range(200):
        method='rank'
        center_set_points,original_point_set,key_center_set,min_gap,key_min,correlated_num,parallel_num,same_time_num=\
            network_algorithm(m_d_set.copy(),point_set.copy(),start_time_set.copy(),duration_set.copy(),
                                severity.copy(),sub_type.copy(),r,threshold=TH,method=method)
        network_cost=cal_cost(same_time_num,r)
        if network_cost[0]<best_result[0]:
            best_result=[network_cost[0],len(center_set_points),sum(same_time_num)]
        if method in ['rank','rank_regulated']:
            break

    # print('linear cost:',best_result[0])
    print('station num:',best_result[1])
    print('drones needed:',best_result[2],'\n')

    # frequency_and_MICSC(center_set_points,key_center_set,correlated_num,parallel_num)
    # draw_figures(min_gap,key_min,correlated_num,parallel_num,same_time_num,r)
    # draw(original_point_set,[p[0] for p in center_set_points],r,[p[1] for p in center_set_points])

    # raster_coverage(center_set_points,key_center_set,raster_area1,'NOAA 2%',r)
    # raster_coverage(center_set_points,key_center_set,raster_area2,'Bob 1991',r)
    # raster_coverage(center_set_points,key_center_set,raster_area3,'Floyd 1999',r)

if __name__ == '__main__':
        read_point_and_process()

