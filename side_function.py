from pyproj import Proj, transform, CRS
#from pyproj import Proj, Transformer
import shapefile
from math import sin, cos, sqrt, atan2, radians, ceil
from datetime import timedelta
# import gdal
import numpy as np
from matplotlib.ticker import PercentFormatter,MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

def covert_coordinate_from_4326_to_MA(point):
    y1, x1 = point[0], point[1]
    # ------original codes, not usable now due to the pyproj package update---
    # inProj = Proj(init='epsg:4326')
    # outProj = Proj(init='epsg:26986')
    # x2, y2 = transform(inProj, outProj, x1, y1)

    # updated codes, note the order of x and y has swapped in the new function
    inProj = CRS('EPSG:4326')
    outProj = CRS('EPSG:26986')
    y2, x2 = transform(inProj, outProj, y1, x1)
    return (y2, x2)

def maintenance_depots(shpname):
    m_d=[]
    sf=shapefile.Reader(shpname)
    shp=sf.shapes()
    length=len(shp)
    # print("\nMaintenance depots: "+str(length))
    for s in shp:
        point=s.points[0]
        output_point=(point[1],point[0])
        m_d.append(output_point)
    return m_d

def find_furthest_point(point,point_set,included_set,r):
    distance=0
    p_f=(0,0)
    for p in point_set:
        if p not in included_set:
            distance_n = cal_distance_simplified(p, point)
            if distance_n > distance:
                distance = distance_n
                p_f = p
    if distance>r:
        return p_f
    return None


def cal_distance_simplified(point_1,point_2):
    return (sqrt((point_1[0]-point_2[0])**2+(point_1[1]-point_2[1])**2))

def find_minimum_time_gap(start_time_set,duration_set,severity,subtype,excluded_set,original_point_set):
    index_set=[]
    for p in excluded_set:
        index_set.append(original_point_set.index(p))
    indexed_incident=[[start_time_set[i],start_time_set[i]+timedelta(minutes=duration_set[i])]for i in index_set]
    for i in range(len(excluded_set)):
        indexed_incident[i]+=[excluded_set[i],severity[i],subtype[i]]
    indexed_incident.sort(key=lambda index_time: index_time[0])
    if len(indexed_incident)==1:
        return 1e8,1e8,1,[],[],([subtype[index_set[0]]],[duration_set[index_set[0]]])


    start_time=[t[0] for t in indexed_incident]
    end_time=[t[1] for t in indexed_incident]
    location=[t[2] for t in indexed_incident]
    severity=[t[3] for t in indexed_incident]
    subtype=[t[4] for t in indexed_incident]

    SIC_duration=[duration_set[i] for i in index_set]
    SIC_severity=severity.copy()

    for i in range(len(subtype)):
        if subtype[i] in ['MVA with injury','multi vehicle accident','MVA without injury']:
            subtype[i]='MVA'
    same_time_threshold=25
    gap_set=[]
    min_gap = timedelta(1e8)
    current_end_time=end_time[0]
    current_occuring=[(start_time[0],end_time[0]+timedelta(minutes=same_time_threshold),location[0],severity[0],subtype[0])]
    correlated_incident_set=[]
    parallel_incident_set=[]

    MIC_set=[]
    for i in range(1,len(indexed_incident)):
        # gap
        gap=max(timedelta(0),start_time[i]-current_end_time)
        gap_set.append(gap)
        if gap<min_gap:
            min_gap=gap
        current_end_time=max(current_end_time,end_time[i])
        # same time incident
        current_occuring = [elem for elem in current_occuring if elem[1]>start_time[i]]
        # same_time_incident=max(same_time_incident,len(current_occuring)+1)
        current_occuring.append((start_time[i],end_time[i]+timedelta(minutes=same_time_threshold),
                                 location[i],severity[i],subtype[i]))
        if len(current_occuring)>1:
            correlated_incidents,parrallel_incidents=find_correlated(current_occuring)

            for full_info in correlated_incidents:
                MIC_set.append(i)
                c_i=full_info
                # c_i=[f[2] for f in full_info]
                if len(correlated_incident_set)==0:
                    correlated_incident_set.append(c_i)
                    continue
                if c_i[0] in correlated_incident_set[-1]:
                    correlated_incident_set.pop(-1)
                correlated_incident_set.append(c_i)

            for p_i in parrallel_incidents:
                if len(parallel_incident_set)==0:
                    parallel_incident_set.append(p_i)
                    continue
                if list(p_i.keys())[0] in list(parallel_incident_set[-1].keys()):
                    parallel_incident_set.pop(-1)
                parallel_incident_set.append(p_i)

    td_mins = round(min_gap.total_seconds() / 60, 1)
    if len(gap_set)>1:
        td_average=np.mean([round(gap.total_seconds() / 60, 1) for gap in gap_set])
    else:
        td_average=round(gap_set[0].total_seconds() / 60, 1)

    try:
        same_time_incident=max([len(p_i) for p_i in parallel_incident_set])
    except:
        same_time_incident = 1

    MIC_set=sorted(list(set(MIC_set)),reverse=True)
    for MIC in MIC_set:
        SIC_duration.pop(MIC)
        SIC_severity.pop(MIC)

    return td_mins,td_average,same_time_incident,correlated_incident_set,parallel_incident_set,(SIC_severity,SIC_duration)


def find_correlated(current_occuring):
    loc=[incident[2] for incident in current_occuring]
    n=len(loc)
    distance_threshold=100
    correlated_incidents=[]
    parallel_incidents=[]

    center=[]
    group={}
    for i in range(n):
        Independence=True
        for c in center:
            if cal_distance_simplified(loc[i],c)<distance_threshold:
                group[c].append(current_occuring[i])
                Independence=False
        if Independence:
           center.append(loc[i])
           group[loc[i]] = [current_occuring[i]]
    for g in group:
        if len(group[g])>1:
            correlated_incidents.append(group[g])
    if len(group)>1:
        parallel_incidents.append(group)
    return correlated_incidents, parallel_incidents

def MIC_analyses(MIC_full_info):
    type_sequence = []
    type_sequence_num = []
    severity = []
    start_time_gap_population = []
    record=[]
    for MIC in MIC_full_info:
        start_time = [i[0] for i in MIC]
        start_time_gap = [round((start_time[i + 1] - start_time[i]).total_seconds() / 60, 1)
                          for i in range(len(start_time) - 1)]
        start_time_gap_population += start_time_gap
        min_start_time = min(start_time)
        end_time = max(i[1] for i in MIC) - timedelta(minutes=25)
        duration = round((end_time - min_start_time).total_seconds() / 60, 1)
        types = [i[4] for i in MIC]
        if types not in type_sequence:
            type_sequence.append(types)
            type_sequence_num.append(1)
        else:
            type_sequence_num[type_sequence.index(types)] += 1
        s = max([int(list((filter(str.isdigit, i[3])))[0]) for i in MIC])
        severity.append((len(types), s, duration))
        record.append((MIC[0][2][0],MIC[0][2][1],len(types),s))
    record.sort(key=lambda ele:int(ele[3]))
    flink = open('MIC.txt','w+')
    for r in record:
        flink.write(
            '%s,%s,%s,%s\n' % (r[0],r[1],r[2],r[3]))
    flink.close()

    subsequence = [(type_sequence[i], type_sequence_num[i], len(type_sequence[i])) for i in range(len(type_sequence))]
    subsequence.sort(key=lambda k: k[1], reverse=True)
    severity_and_duration(severity)


def severity_and_duration(severity):
    severity_dic={}
    for i in severity:
        num_of_inc=i[0]
        if num_of_inc in severity_dic.keys():
            severity_dic[num_of_inc].append((i[1],i[2]))
        else:
            severity_dic[num_of_inc]=[(i[1],i[2])]
    num_of_level2=[sum([1 for i in severity_dic[2] if i[0] == 2]),
     sum([1 for i in severity_dic[3] if i[0] == 2]),
     sum([1 for i in severity_dic[4] if i[0] == 2])]
    num_of_level3 =[sum([1 for i in severity_dic[2] if i[0] == 3]),
     sum([1 for i in severity_dic[3] if i[0] == 3]),
     sum([1 for i in severity_dic[4] if i[0] == 3])]
    mean_of_duration=[np.mean([i[1] for i in severity_dic[2]]),
     np.mean([i[1] for i in severity_dic[3]]),
     np.mean([i[1] for i in severity_dic[4]])]


def circle_rank(center_set,point_set,r):
    circle_info=[]
    max_objective=0
    for center in center_set:
        included_set=find_included(center,point_set,[],r)
        if len(included_set)==0:
            continue
        objective=len(included_set)
        if objective>max_objective:
            circle_info=(center,included_set)
            max_objective=objective
    print(len(included_set))
    return circle_info

def circle_rank_regulated(center_set,point_set,r,start_time_set,duration_set,severity,sub_type,original_point_set,alpha):
    circle_info=[]
    max_objective=1e8
    for center in center_set:
        print(center)
        included_set=find_included(center,point_set,[],r)
        if len(included_set)==0:
            continue
        min_t_gap, avg_t_gap, num_same_time_incident, correlated_incidents, parallel_incidents, SIC_info = find_minimum_time_gap \
            (start_time_set, duration_set, severity, sub_type, included_set, original_point_set)
        single_station_cost=cal_single_cost(num_same_time_incident,r)
        # single_station_cost=(1e8,1e8)
        # objective=len(included_set)-alpha*single_station_cost[0]
        objective=single_station_cost[0]/len(included_set)

        if objective<max_objective:
            circle_info=(center,included_set)
            max_objective=objective
            b=(min_t_gap, avg_t_gap, num_same_time_incident, correlated_incidents, parallel_incidents,SIC_info)
    if len(circle_info)==0:
        return circle_info,0,0,0,0,0,([],[])
    return circle_info,b[0], b[1], b[2], b[3], b[4],b[5]

def circle_random(center_set,point_set,r,start_time_set,duration_set,severity,sub_type,original_point_set):
    center=random.choice(center_set)
    included_set=find_included(center,point_set,[],r)
    circle_info=(center,included_set)
    min_t_gap, avg_t_gap, num_same_time_incident, correlated_incidents, parallel_incidents = find_minimum_time_gap \
        (start_time_set, duration_set, severity, sub_type, included_set, original_point_set)
    return circle_info,min_t_gap, avg_t_gap, num_same_time_incident, correlated_incidents, parallel_incidents


def find_included(point, point_set, included_set,distance):
    for p in point_set:
        if p not in included_set:
            distance_n=cal_distance_simplified(p, point)
            if distance_n<=distance:
                included_set.append(p)
    return included_set


def read_raster(raster_name):
    ds = gdal.Open(raster_name)
    cols=ds.RasterXSize
    rows=ds.RasterYSize
    geo= ds.GetGeoTransform()
    originX=geo[0]
    originY=geo[3]
    pixelWidth=geo[1]
    pixelHeight=geo[5]
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray(0, 0, cols, rows)
    point_set=[]
    for c in range(cols):
        for r in range(rows):
            if data[r,c]==1:
                point=cal_coor_from_offset(c,r,originX,originY,pixelWidth,pixelHeight)
                point_set.append(point)
    return point_set



def cal_coor_from_offset(xoffset,yoffset,originX,originY,pixelWidth,pixelHeight):
    x = xoffset * pixelWidth + originX
    y = yoffset * pixelHeight + originY
    return (y,x)



def cal_cost(same_time_num,r):
    r_in_mile=r / 1000 / 1.60934

    cost_linear=0
    alpha=0.5
    b=1
    individual_cost_linear=r_in_mile*alpha+b

    cost_exponential=0
    base=1.25
    individual_cost_exponential=base**r_in_mile

    for num in same_time_num:
        cost_linear+=num*individual_cost_linear
        cost_exponential+=num*individual_cost_exponential

    return (round(cost_linear,1),round(cost_exponential,1))


def cal_single_cost(num,r):
    r_in_mile=r / 1000 / 1.60934
    alpha=0.5
    b=1
    individual_cost_linear=r_in_mile*alpha+b
    base=1.25
    individual_cost_exponential=base**r_in_mile
    cost_linear=num*individual_cost_linear
    cost_exponential=num*individual_cost_exponential

    return (round(cost_linear,1),round(cost_exponential,1))


def raster_coverage(center_point_set,key_station,point_set,name,r):
    included_set=[]
    key_coverage=0
    for c_p in center_point_set:
        old_included_set=included_set.copy()
        included_set=find_included(c_p[0],point_set,included_set,r)
        if c_p in key_station:
            key_coverage+=len(included_set)-len(old_included_set)
    coverage=len(included_set)/len(point_set)
    k_c=key_coverage/len(point_set)
    print('%s converage:'%name,coverage,' key coverage:',k_c)




def generate_random_circle(point_set,n,r):
    center_set=[]
    for point in point_set:
        for i in range(n):
            center_x=point[0]+random.uniform(-1,1)*r/100
            center_y=point[1]+random.uniform(-1,1)*r/100
            center_set.append((center_x,center_y))
    return center_set

def sortSecondLen(val):
    return len(val[1])

def draw(point_set,center_set,r,frequency=None):
    fig = plt.figure(figsize=(13, 6), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    scatter1=[p[0] for p in point_set]
    scatter2=[p[1] for p in point_set]
    plt.scatter(scatter2,scatter1,s=1,edgecolors='k')
    for i in range(len(center_set)):
        p=center_set[i]
        f=frequency[i]
        if frequency is not None:
            c,a=decide_color(f,threshold=120)
        patch = patches.Circle((p[1],p[0]), radius=r,fc=c,alpha=a)
        ax.add_patch(patch)
    plt.legend((patches.Circle((0,0), radius=r,fc='r',alpha=0.2),patches.Circle((0,0), radius=r,fc='g',alpha=0.1)), ('Key stations', 'Regular Stations'))
    plt.savefig('station.png')

def decide_color(f,threshold):
    if f>=threshold:
        return 'r',0.2
    else:
        return 'g',0.1

def get_between(x, d):
    for i in range(len(x)):
        if d<=x[i+1]:
            return i

def greedy_generate(point_set,r):
    original_point_set=point_set.copy()
    final_center_set=[]
    while len(point_set)>len(original_point_set)*0.3:
        print(len(point_set))
        center_set = generate_random_circle(point_set, 1,r)
        circle_info=circle_rank(center_set,point_set)
        final_center_set.append(circle_info[0])
        excluded_set=circle_info[1]
        excluded_set.sort(reverse=True)
        for point in excluded_set:
            point_set.pop(point)
    print(len(final_center_set))
    draw(original_point_set,final_center_set)

def draw_figures(min_gap,key_min,correlated_num,parallel_num,same_time_num,r):
    start = 0
    end = 125
    bin = int((end - start) / 5)
    min_gap_duration_groups = [0 for i in range(5)]
    for a_g in min_gap:
        d = get_between(np.append(np.arange(start, end, bin), [1e8]), a_g)
        min_gap_duration_groups[d] += 1
    fig = plt.figure(figsize=(7, 4.5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    labels = [f / len(min_gap) for f in min_gap_duration_groups]
    rects = ax.bar(np.arange(start + bin / 2, end, bin), labels, width=bin)
    plt.xticks(np.arange(start + bin / 2, end, bin),
               ['%s - %s' % (start, start + bin), '%s - %s' % (start + bin, start + bin * 2),
                '%s - %s' % (start + bin * 2, start + bin * 3), '%s - %s' % (start + bin * 3, start + bin * 4),
                'more than\n %s' % (start + bin * 4)], fontsize=16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    axes = plt.gca()
    y_limit = axes.get_ylim()
    coef = [1, 1.1]
    y_limit = [y_limit[i] * coef[i] for i in range(2)]
    axes.set_ylim(y_limit)
    axes.set_xlim(0, end)
    axes.yaxis.set_major_formatter(PercentFormatter(1))
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        get_x = rect.get_x()
        ax.text(get_x + bin / 2, height + y_limit[1] * 0.04,
                "{0:.1f}%".format(labels[i] * 100), ha='center', va='center', fontsize=18)
    plt.title('All stations', fontsize=16)
    plt.xlabel('minimum time gap (min)', fontsize=16)
    plt.savefig('min time gap.png')
    plt.close()

    min_gap = key_min
    min_gap_duration_groups = [0 for i in range(5)]
    for a_g in min_gap:
        d = get_between(np.append(np.arange(start, end, bin), [1e8]), a_g)
        min_gap_duration_groups[d] += 1
    fig = plt.figure(figsize=(7, 4.5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    labels = [f / len(min_gap) for f in min_gap_duration_groups]
    rects = ax.bar(np.arange(start + bin / 2, end, bin), labels, width=bin)
    plt.xticks(np.arange(start + bin / 2, end, bin),
               ['%s - %s' % (start, start + bin), '%s - %s' % (start + bin, start + bin * 2),
                '%s - %s' % (start + bin * 2, start + bin * 3), '%s - %s' % (start + bin * 3, start + bin * 4),
                'more than\n %s' % (start + bin * 4)], fontsize=16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    axes = plt.gca()
    y_limit = axes.get_ylim()
    coef = [1, 1.1]
    y_limit = [y_limit[i] * coef[i] for i in range(2)]
    axes.set_ylim(y_limit)
    axes.set_xlim(0, end)
    axes.yaxis.set_major_formatter(PercentFormatter(1))
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        get_x = rect.get_x()
        ax.text(get_x + bin / 2, height + y_limit[1] * 0.04,
                "{0:.1f}%".format(labels[i] * 100), ha='center', va='center', fontsize=18)
    plt.title('Key stations', fontsize=16)
    plt.xlabel('minimum time gap (min)', fontsize=16)
    plt.savefig('key min time gap.png')
    plt.close()

    start = 1
    end = 6
    bin = int((end - start + 1) / 3)
    hist_range = np.append([0], np.append(np.arange(start - 0.5, end + 1, bin), [1e8]))
    min_gap = correlated_num
    min_gap_duration_groups = [0 for i in range(5)]
    for a_g in min_gap:
        d = get_between(hist_range, a_g)
        min_gap_duration_groups[d] += 1
    fig = plt.figure(figsize=(7, 4.5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    labels = [f / len(min_gap) for f in min_gap_duration_groups]
    rects = ax.bar(np.arange(1, 6, 1), labels, width=1)
    plt.xticks(np.arange(1, 6, 1),
               ['0', '%s - %s' % (start, start + bin - 1), '%s - %s' % (start + bin, start + bin * 2 - 1),
                '%s - %s' % (start + bin * 2, start + bin * 3 - 1),
                'more than\n %s' % (start + bin * 3 - 1)], fontsize=16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    axes = plt.gca()
    y_limit = axes.get_ylim()
    coef = [1, 1.1]
    y_limit = [y_limit[i] * coef[i] for i in range(2)]
    axes.set_ylim(y_limit)
    axes.set_xlim(0.5, 5.5)
    axes.yaxis.set_major_formatter(PercentFormatter(1))
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        get_x = rect.get_x()
        ax.text(get_x + 1 / 2, height + y_limit[1] * 0.04,
                "{0:.1f}%".format(labels[i] * 100), ha='center', va='center', fontsize=18)
    plt.title('Radius = %s miles' % (int(r / 1000 / 1.60934)), fontsize=16)
    plt.xlabel('Number of correlated incident groups', fontsize=16)
    plt.savefig('correlated incident.png')
    plt.close()

    min_gap = parallel_num
    min_gap_duration_groups = [0 for i in range(5)]
    for a_g in min_gap:
        d = get_between(hist_range, a_g)
        min_gap_duration_groups[d] += 1
    fig = plt.figure(figsize=(7, 4.5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    labels = [f / len(min_gap) for f in min_gap_duration_groups]
    rects = ax.bar(np.arange(1, 6, 1), labels, width=1)
    plt.xticks(np.arange(1, 6, 1),
               ['0', '%s - %s' % (start, start + bin - 1), '%s - %s' % (start + bin, start + bin * 2 - 1),
                '%s - %s' % (start + bin * 2, start + bin * 3 - 1),
                'more than\n %s' % (start + bin * 3 - 1)], fontsize=16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    axes = plt.gca()
    y_limit = axes.get_ylim()
    coef = [1, 1.1]
    y_limit = [y_limit[i] * coef[i] for i in range(2)]
    axes.set_ylim(y_limit)
    axes.set_xlim(0.5, 5.5)
    axes.yaxis.set_major_formatter(PercentFormatter(1))
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        get_x = rect.get_x()
        ax.text(get_x + 1 / 2, height + y_limit[1] * 0.04,
                "{0:.1f}%".format(labels[i] * 100), ha='center', va='center', fontsize=18)
    plt.title('Radius = %s miles' % (int(r / 1000 / 1.60934)), fontsize=16)
    plt.xlabel('Number of parallel incident groups', fontsize=16)
    plt.savefig('parallel incident.png')
    plt.close()

    min_gap = [correlated_num[i] + parallel_num[i] for i in range(len(correlated_num))]
    min_gap_duration_groups = [0 for i in range(5)]
    for a_g in min_gap:
        d = get_between(hist_range, a_g)
        min_gap_duration_groups[d] += 1
    fig = plt.figure(figsize=(7, 4.5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    labels = [f / len(min_gap) for f in min_gap_duration_groups]
    rects = ax.bar(np.arange(1, 6, 1), labels, width=1)
    plt.xticks(np.arange(1, 6, 1),
               ['0', '%s - %s' % (start, start + bin - 1), '%s - %s' % (start + bin, start + bin * 2 - 1),
                '%s - %s' % (start + bin * 2, start + bin * 3 - 1),
                'more than\n %s' % (start + bin * 3 - 1)], fontsize=16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    axes = plt.gca()
    y_limit = axes.get_ylim()
    coef = [1, 1.1]
    y_limit = [y_limit[i] * coef[i] for i in range(2)]
    axes.set_ylim(y_limit)
    axes.set_xlim(0.5, 5.5)
    axes.yaxis.set_major_formatter(PercentFormatter(1))
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        get_x = rect.get_x()
        ax.text(get_x + 1 / 2, height + y_limit[1] * 0.04,
                "{0:.1f}%".format(labels[i] * 100), ha='center', va='center', fontsize=18)
    plt.title('Radius = %s miles' % (int(r / 1000 / 1.60934)), fontsize=16)
    plt.xlabel('Number of incident groups\n ocurring at the same time at each station', fontsize=16)
    plt.savefig('total incident.png')
    plt.close()

    same_time_incident_groups = [0 for i in range(4)]
    for a_g in same_time_num:
        d = get_between(np.arange(0.5, 5, 1), a_g)
        same_time_incident_groups[d] += 1
    fig = plt.figure(figsize=(7, 4.5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)

    labels = [f / len(same_time_num) for f in same_time_incident_groups]
    labels = same_time_incident_groups

    rects = ax.bar(np.arange(1, 5, 1), labels, width=1)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    axes = plt.gca()
    y_limit = axes.get_ylim()
    coef = [1, 1.1]
    y_limit = [y_limit[i] * coef[i] for i in range(2)]
    axes.set_ylim(y_limit)
    axes.set_xlim(0.5, 4.5)
    # axes.yaxis.set_major_formatter(PercentFormatter(1))
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        get_x = rect.get_x()
        # ax.text(get_x + 1/2, height + y_limit[1]*0.04, "{0:.1f}%".format(labels[i] * 100), ha='center', va='center',fontsize=18)
        ax.text(get_x + 1 / 2, height + y_limit[1] * 0.04, labels[i], ha='center', va='center', fontsize=18)
    plt.title('Radius = %s miles' % (int(r / 1000 / 1.60934)), fontsize=16)
    plt.xlabel('Highest number of parallel incidents\n occurring at the same time', fontsize=16)
    plt.savefig('same time incident.png')
    plt.close()

def frequency_and_MICSC(center_set_points,key_center_set,correlated_num,parallel_num):
    y=correlated_num
    # y=parallel_num

    fig = plt.figure(figsize=(5, 5), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    frequency = [f[1] for f in center_set_points]
    frequency_center = [c[1] for c in key_center_set]
    correlated_key = [y[i] for i in range(len(frequency_center))]
    plt.scatter(frequency, y, c='b', label='regular station', s=48, edgecolor='k')
    plt.scatter(frequency_center, correlated_key, c='r', label='key station', s=48, edgecolor='k')
    plt.legend()
    plt.xlabel('Number of incidents', fontsize=16)
    plt.ylabel('Number of MICs', fontsize=16)
    r_in_mile = int(r / 1000 / 1.60934)
    plt.title('Radius = ' + str(r_in_mile) + ' miles')
    axes = plt.gca()
    y_limit = axes.get_ylim()
    y_limit = [-0.5, ceil(y_limit[1] / 5) * 5]
    axes.set_ylim(y_limit)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('relationship.png')

def cal_distance(point_1,point_2):
    # approximate radius of earth in km
    R = 6373.0
    lat1 = radians(point_1[0])
    lon1 = radians(point_1[1])
    lat2 = radians(point_2[0])
    lon2 = radians(point_2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def linear_regression(X,Y):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(X, Y)
    # Make predictions using the testing set
    y_pred = regr.predict(X)

    newX = np.append(np.ones((len(X),1)), X, axis=1)
    MSE = (sum((Y-y_pred)**2))/(len(newX)-len(newX[0]))
    params = np.append(regr.intercept_, regr.coef_)

    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]
    sd_b = np.round(sd_b, 3)
    ts_b = np.round(ts_b, 3)
    params = np.round(params, 4)
    print('Intercept and Coefficients: \n', params)
    print("p value:",p_values)

    return params

    # Plot outputs
    # plt.scatter(X, Y, color='black')
    # plt.plot(X, y_pred, color='blue', linewidth=3)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()

def network_algorithm(center_set,point_set,start_time_set,duration_set,severity,sub_type,r,threshold,method):
    key_threshold = r / 1000 / 1.60934 * 12
    original_point_set = point_set.copy()
    final_center_set = []
    key_sum=0
    key_count=0
    key_center_set=[]
    min_t_gap_set=[]
    average_t_gap_set=[]
    key_min_t_gap_set=[]
    key_average_t_gap_set=[]
    correlated_incidents_set=[]
    parallel_incident_set=[]
    same_time_incident_set=[]
    MIC_full_info=[]
    SIC_full_info=[[],[]]
    circle_info=[]
    flink = open('stations.txt','w+')

    while len(point_set)>(len(original_point_set)*threshold):
        # rank
        if method in ['rank','rank_regulated']:
            regulation_factor = 0
            circle_info,min_t_gap, avg_t_gap, num_same_time_incident, correlated_incidents, parallel_incidents,SIC_info\
                = circle_rank_regulated(center_set, point_set,r,start_time_set,duration_set,severity,sub_type,original_point_set.copy(),
                              alpha=regulation_factor)
            if len(circle_info)==0:
                break
        #random
        if method=='random':
            if len(circle_info)>0:
                center_set.remove(circle_info[0])
            if not len(center_set):
                break
            circle_info = circle_random(center_set,point_set,r,start_time_set,duration_set,severity,sub_type,original_point_set)
            if not len(circle_info):
                continue

        excluded_set=circle_info[1]
        final_center_set.append((circle_info[0],len(excluded_set)))

        MIC_full_info+=correlated_incidents
        SIC_full_info=[SIC_full_info[i]+SIC_info[i] for i in range(2)]
        min_t_gap_set.append(min_t_gap)
        average_t_gap_set.append(avg_t_gap)
        correlated_incidents_set.append(len(correlated_incidents))
        parallel_incident_set.append(len(parallel_incidents))
        same_time_incident_set.append(num_same_time_incident)
        max_cov=len(excluded_set)
        if max_cov>=key_threshold:
            key_center_set.append((circle_info[0], len(excluded_set)))
            key_sum+=max_cov
            key_count+=1
            key_min_t_gap_set.append(min_t_gap)
            key_average_t_gap_set.append(avg_t_gap)
            if max_cov>=600:
                flink.write('%s,%s,%s,%s,%s,%s\n' % (
                circle_info[0][0], circle_info[0][1], len(circle_info[1]), min_t_gap, 'c', num_same_time_incident))
            else:
                flink.write('%s,%s,%s,%s,%s,%s\n' % (
                    circle_info[0][0], circle_info[0][1], len(circle_info[1]), min_t_gap, 'k', num_same_time_incident))
        else:
            flink.write('%s,%s,%s,%s,%s,%s\n' % (
                circle_info[0][0], circle_info[0][1], len(circle_info[1]), min_t_gap, 'r', num_same_time_incident))

        excluded_set.sort(reverse=True)
        for point in excluded_set:
            point_set.remove(point)

    # MIC_analyses(MIC_full_info)

    return final_center_set,original_point_set,key_center_set,\
           min_t_gap_set,key_min_t_gap_set,correlated_incidents_set,parallel_incident_set, same_time_incident_set
