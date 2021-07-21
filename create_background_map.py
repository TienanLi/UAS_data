from side_function import covert_coordinate_from_4326_to_MA
import shapefile
import pickle
import numpy as np

def transfer_polygon(points):
    new_points = []
    for p in points:
        y_T, x_T = covert_coordinate_from_4326_to_MA((float(p[1]), float(p[0])))
        new_points.append((x_T, y_T))
    return new_points

def simplify_and_transfer_polygon(points, max_points = None):
    if not max_points:
        return transfer_polygon(points)

    new_points = []
    if len(points) > max_points:
        new_points.append(points[0])
        for i in np.random.choice(np.arange(1, len(points)), max_points - 2):
            new_points.append(points[i])
        new_points.append(points[-1])
        return new_points
    else:
        return transfer_polygon(points)

def main():
    motorway_list = []
    print('putting on road_map...')
    sf=shapefile.Reader('gis_osm_roads_free_1.shp')
    shp=sf.shapeRecords()
    for i in range(len(shp)):
        s = shp[i]
        if s.record.fclass == 'motorway':
            print(i)
            motorway_list.append(simplify_and_transfer_polygon(s.shape.points))
    print('finish putting on road_map')

    filehandler = open('road_map.obj', 'wb')
    pickle.dump(motorway_list, filehandler)
    del motorway_list

if __name__ == '__main__':
    main()