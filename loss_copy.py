
def loss_total(output_gather,labels_gather):

    loss_normal = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_gather, labels=labels_gather)
    ###chair
    labels_chair = tf.equal(labels_gather,9)
    pixel_inds_chair = tf.squeeze(tf.where(labels_chair), 1)
    labels_chair_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_chair))
    output_chair = tf.gather(output_gather, pixel_inds_chair)
    assert output_chair.get_shape().as_list()[0] == labels_chair_gather.get_shape().as_list()[0]
    output_chair = tf.nn.softmax(output_chair)
    output_chair_table = tf.identity(output_chair[:,11])
    output_chair_table_loss_inds = tf.less(output_chair_table,0.9) #and tf.greater(output_chair_table,0.2)
    output_chair_table_loss_inds = tf.squeeze(tf.where(output_chair_table_loss_inds), 1)
    output_chair_table_loss = tf.gather(output_chair_table,output_chair_table_loss_inds)
    output_chair_table_loss = 1-output_chair_table_loss
    output_chair_table_loss = -tf.log(output_chair_table_loss)
    loss_chair_table = tf.reduce_mean(output_chair_table_loss)
    loss_chair_table = tf.cond(tf.equal(tf.is_nan(loss_chair_table),False),lambda:tf.identity(loss_chair_table),lambda:tf.identity(0.))


    output_chair_background = output_chair[:,0]
    output_chair_background_loss_inds = tf.less(output_chair_background,0.9)
    output_chair_background_loss_inds = tf.squeeze(tf.where(output_chair_background_loss_inds), 1)
    output_chair_background_loss = tf.gather(output_chair_background,output_chair_background_loss_inds)
    output_chair_background_loss = 1-output_chair_background_loss
    output_chair_background_loss = tf.log(output_chair_background_loss)
    loss_chair_background = -tf.reduce_mean(output_chair_background_loss)
    loss_chair_background = tf.cond(tf.equal(tf.is_nan(loss_chair_background),False),lambda:tf.identity(loss_chair_background),lambda:tf.identity(0.))
    chair_loss = loss_chair_table + loss_chair_background

    ###table
    labels_table = tf.equal(labels_gather,11)
    pixel_inds_table = tf.squeeze(tf.where(labels_table), 1)
    labels_table_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_table))
    output_table = tf.gather(output_gather, pixel_inds_table)
    assert output_table.get_shape().as_list()[0] == labels_table_gather.get_shape().as_list()[0]
    output_table = tf.nn.softmax(output_table)

    output_table_chair = tf.identity(output_table[:,9])
    output_table_chair_loss_inds = tf.less(output_table_chair,0.9) #and tf.greater(output_chair_table,0.2)
    output_table_chair_loss_inds = tf.squeeze(tf.where(output_table_chair_loss_inds), 1)
    output_table_chair_loss = tf.gather(output_table_chair,output_table_chair_loss_inds)
    output_table_chair = 1 - output_table_chair_loss
    output_table_chair = tf.log(output_table_chair)
    loss_table_chair = -tf.reduce_mean(output_table_chair)
    loss_table_chair = tf.cond(tf.equal(tf.is_nan(loss_table_chair),False),lambda:tf.identity(loss_table_chair),lambda:tf.identity(0.))

    output_table_background = output_table[:,0]
    output_table_background_loss_inds = tf.less(output_table_background,0.9) #and tf.greater(output_table_background,0.2)
    output_table_background_loss_inds = tf.squeeze(tf.where(output_table_background_loss_inds), 1)
    output_table_background_loss = tf.gather(output_table_background,output_table_background_loss_inds)
    output_table_background_loss = 1-output_table_background_loss
    output_table_background_loss = tf.log(output_table_background_loss)
    loss_table_background = -tf.reduce_mean(output_table_background_loss)
    loss_table_background = tf.cond(tf.equal(tf.is_nan(loss_table_background),False),lambda:tf.identity(loss_table_background),lambda:tf.identity(0.))
    table_loss = loss_table_chair + loss_table_background


    ###bike
    labels_bike = tf.equal(labels_gather,2)
    pixel_inds_bike = tf.squeeze(tf.where(labels_bike), 1)
    labels_bike_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_bike))
    output_bike = tf.gather(output_gather, pixel_inds_bike)
    assert output_bike.get_shape().as_list()[0] == labels_bike_gather.get_shape().as_list()[0]
    output_bike = tf.nn.softmax(output_bike)
    output_bike_background = tf.identity(output_bike[:,0])
    output_bike_background_loss_inds = tf.less(output_bike_background,0.9) #and tf.greater(output_background_bike,0.2)
    output_bike_background_loss_inds = tf.squeeze(tf.where(output_bike_background_loss_inds), 1)
    output_bike_background_loss = tf.gather(output_bike_background,output_bike_background_loss_inds)
    output_bike_background = 1 - output_bike_background_loss
    output_bike_background = tf.log(output_bike_background)
    loss_bike_background = -tf.reduce_mean(output_bike_background)
    loss_bike_background = tf.cond(tf.equal(tf.is_nan(loss_bike_background),False),lambda:tf.identity(loss_bike_background),lambda:tf.identity(0.))
    bike_loss = loss_bike_background
    ###boat
    labels_boat = tf.equal(labels_gather,4)
    pixel_inds_boat = tf.squeeze(tf.where(labels_boat), 1)
    labels_boat_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_boat))
    output_boat = tf.gather(output_gather, pixel_inds_boat)
    assert output_boat.get_shape().as_list()[0] == labels_boat_gather.get_shape().as_list()[0]
    output_boat = tf.nn.softmax(output_boat)

    output_boat_car = tf.identity(output_boat[:,7])
    output_boat_car_loss_inds = tf.less(output_boat_car,0.9) #and tf.greater(output_car_boat,0.2)
    output_boat_car_loss_inds = tf.squeeze(tf.where(output_boat_car_loss_inds), 1)
    output_boat_car_loss = tf.gather(output_boat_car,output_boat_car_loss_inds)
    output_boat_car = 1 - output_boat_car_loss
    output_boat_car = tf.log(output_boat_car)
    loss_boat_car = -tf.reduce_mean(output_boat_car)
    loss_boat_car = tf.cond(tf.equal(tf.is_nan(loss_boat_car),False),lambda:tf.identity(loss_boat_car),lambda:tf.identity(0.))

    output_boat_plane = tf.identity(output_boat[:,1])
    output_boat_plane_loss_inds = tf.less(output_boat_plane,0.9) #and tf.greater(output_plane_boat,0.2)
    output_boat_plane_loss_inds = tf.squeeze(tf.where(output_boat_plane_loss_inds), 1)
    output_boat_plane_loss = tf.gather(output_boat_plane,output_boat_plane_loss_inds)
    output_boat_plane = 1 - output_boat_plane_loss
    output_boat_plane = tf.log(output_boat_plane)
    loss_boat_plane = -tf.reduce_mean(output_boat_plane)
    loss_boat_plane = tf.cond(tf.equal(tf.is_nan(loss_boat_plane),False),lambda:tf.identity(loss_boat_plane),lambda:tf.identity(0.))

    output_boat_background = tf.identity(output_boat[:,0])
    output_boat_background_loss_inds = tf.less(output_boat_background,0.9) #and tf.greater(output_background_boat,0.2)
    output_boat_background_loss_inds = tf.squeeze(tf.where(output_boat_background_loss_inds), 1)
    output_boat_background_loss = tf.gather(output_boat_background,output_boat_background_loss_inds)
    output_boat_background = 1 - output_boat_background_loss
    output_boat_background = tf.log(output_boat_background)
    loss_boat_background = -tf.reduce_mean(output_boat_background)
    loss_boat_background = tf.cond(tf.equal(tf.is_nan(loss_boat_background),False),lambda:tf.identity(loss_boat_background),lambda:tf.identity(0.))
    boat_loss = 0.5*loss_boat_car+0.5*loss_boat_plane+loss_boat_background
    ###plant
    labels_plant = tf.equal(labels_gather,16)
    pixel_inds_plant = tf.squeeze(tf.where(labels_plant), 1)
    labels_plant_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_plant))
    output_plant = tf.gather(output_gather, pixel_inds_plant)
    assert output_plant.get_shape().as_list()[0] == labels_plant_gather.get_shape().as_list()[0]
    output_plant = tf.nn.softmax(output_plant)
    output_plant_background = tf.identity(output_plant[:,0])
    output_plant_background_loss_inds = tf.less(output_plant_background,0.9) #and tf.greater(output_background_plant,0.2)
    output_plant_background_loss_inds = tf.squeeze(tf.where(output_plant_background_loss_inds), 1)
    output_plant_background_loss = tf.gather(output_plant_background,output_plant_background_loss_inds)
    output_plant_background = 1 - output_plant_background_loss
    output_plant_background = tf.log(output_plant_background)
    loss_plant_background = -tf.reduce_mean(output_plant_background)
    loss_plant_background = tf.cond(tf.equal(tf.is_nan(loss_plant_background),False),lambda:tf.identity(loss_plant_background),lambda:tf.identity(0.))
    plant_loss = loss_plant_background
    ###sofa
    labels_sofa = tf.equal(labels_gather,18)
    pixel_inds_sofa = tf.squeeze(tf.where(labels_sofa), 1)
    labels_sofa_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_sofa))
    output_sofa = tf.gather(output_gather, pixel_inds_sofa)
    assert output_sofa.get_shape().as_list()[0] == labels_sofa_gather.get_shape().as_list()[0]
    output_sofa = tf.nn.softmax(output_sofa)

    output_sofa_person = tf.identity(output_sofa[:,15])
    output_sofa_person_loss_inds = tf.less(output_sofa_person,0.9) #and tf.greater(output_person_sofa,0.2)
    output_sofa_person_loss_inds = tf.squeeze(tf.where(output_sofa_person_loss_inds), 1)
    output_sofa_person_loss = tf.gather(output_sofa_person,output_sofa_person_loss_inds)
    output_sofa_person = 1 - output_sofa_person_loss
    output_sofa_person = tf.log(output_sofa_person)
    loss_sofa_person = -tf.reduce_mean(output_sofa_person)
    loss_sofa_person = tf.cond(tf.equal(tf.is_nan(loss_sofa_person),False),lambda:tf.identity(loss_sofa_person),lambda:tf.identity(0.))

    output_sofa_dog = tf.identity(output_sofa[:,12])
    output_sofa_dog_loss_inds = tf.less(output_sofa_dog,0.9) #and tf.greater(output_dog_sofa,0.2)
    output_sofa_dog_loss_inds = tf.squeeze(tf.where(output_sofa_dog_loss_inds), 1)
    output_sofa_dog_loss = tf.gather(output_sofa_dog,output_sofa_dog_loss_inds)
    output_sofa_dog = 1 - output_sofa_dog_loss
    output_sofa_dog = tf.log(output_sofa_dog)
    loss_sofa_dog = -tf.reduce_mean(output_sofa_dog)
    loss_sofa_dog = tf.cond(tf.equal(tf.is_nan(loss_sofa_dog),False),lambda:tf.identity(loss_sofa_dog),lambda:tf.identity(0.))

    output_sofa_cat = tf.identity(output_sofa[:,8])
    output_sofa_cat_loss_inds = tf.less(output_sofa_cat,0.9) #and tf.greater(output_cat_sofa,0.2)
    output_sofa_cat_loss_inds = tf.squeeze(tf.where(output_sofa_cat_loss_inds), 1)
    output_sofa_cat_loss = tf.gather(output_sofa_cat,output_sofa_cat_loss_inds)
    output_sofa_cat = 1 - output_sofa_cat_loss
    output_sofa_cat = tf.log(output_sofa_cat)
    loss_sofa_cat = -tf.reduce_mean(output_sofa_cat)
    loss_sofa_cat = tf.cond(tf.equal(tf.is_nan(loss_sofa_cat),False),lambda:tf.identity(loss_sofa_cat),lambda:tf.identity(0.))

    output_sofa_background = tf.identity(output_sofa[:,0])
    output_sofa_background_loss_inds = tf.less(output_sofa_background,0.9) #and tf.greater(output_background_sofa,0.2)
    output_sofa_background_loss_inds = tf.squeeze(tf.where(output_sofa_background_loss_inds), 1)
    output_sofa_background_loss = tf.gather(output_sofa_background,output_sofa_background_loss_inds)
    output_sofa_background = 1 - output_sofa_background_loss
    output_sofa_background = tf.log(output_sofa_background)
    loss_sofa_background = -tf.reduce_mean(output_sofa_background)
    loss_sofa_background = tf.cond(tf.equal(tf.is_nan(loss_sofa_background),False),lambda:tf.identity(loss_sofa_background),lambda:tf.identity(0.))
    sofa_loss = 0.5*loss_sofa_person+0.5*loss_sofa_dog+0.5*loss_sofa_cat+0.5*loss_sofa_background


    ###tv
    labels_tv = tf.equal(labels_gather,20)
    pixel_inds_tv = tf.squeeze(tf.where(labels_tv), 1)
    labels_tv_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_tv))
    output_tv = tf.gather(output_gather, pixel_inds_tv)
    assert output_tv.get_shape().as_list()[0] == labels_tv_gather.get_shape().as_list()[0]
    output_tv = tf.nn.softmax(output_tv)
    output_tv_background = tf.identity(output_tv[:,0])
    output_tv_background_loss_inds = tf.less(output_tv_background,0.9) #and tf.greater(output_background_tv,0.2)
    output_tv_background_loss_inds = tf.squeeze(tf.where(output_tv_background_loss_inds), 1)
    output_tv_background_loss = tf.gather(output_tv_background,output_tv_background_loss_inds)
    output_tv_background = 1 - output_tv_background_loss
    output_tv_background = tf.log(output_tv_background)
    loss_tv_background = -tf.reduce_mean(output_tv_background)
    loss_tv_background = tf.cond(tf.equal(tf.is_nan(loss_tv_background),False),lambda:tf.identity(loss_tv_background),lambda:tf.identity(0.))
    tv_loss = loss_tv_background
    inter_class_loss = tf.add_n([chair_loss,table_loss,bike_loss,boat_loss,plant_loss,sofa_loss,tv_loss])

    ###intra_class promotion

    output_gather = tf.nn.softmax(output_gather)
    labels_plane = tf.equal(labels_gather,1)
    pixel_inds_plane = tf.squeeze(tf.where(labels_plane), 1)
    output_plane = tf.gather(output_gather, pixel_inds_plane)
    output_plane = output_plane[:,1]
    plane_mean, plane_var = tf.nn.moments(output_plane,axes=[0])
    #plane_var = tf.cond(tf.equal(tf.is_nan(plane_var),False),lambda:tf.identity(plane_var),lambda:tf.identity(0.))
    plane_abs = tf.abs(output_plane-plane_mean)
    plane_abs = tf.reduce_mean(plane_abs)
    plane_abs = tf.cond(tf.equal(tf.is_nan(plane_abs),False),lambda:tf.identity(plane_abs),lambda:tf.identity(0.))

    #plane_var = tf.sqrt(plane_var)

    labels_bike = tf.equal(labels_gather,2)
    pixel_inds_bike = tf.squeeze(tf.where(labels_bike), 1)
    output_bike = tf.gather(output_gather, pixel_inds_bike)
    output_bike = output_bike[:,2]
    bike_mean, bike_var = tf.nn.moments(output_bike,axes=[0])
    bike_var = tf.cond(tf.equal(tf.is_nan(bike_var),False),lambda:tf.identity(bike_var),lambda:tf.identity(0.))
    #bike_var = tf.sqrt(bike_var)

    labels_bird = tf.equal(labels_gather,3)
    pixel_inds_bird = tf.squeeze(tf.where(labels_bird), 1)
    output_bird = tf.gather(output_gather, pixel_inds_bird)
    output_bird = output_bird[:,3]
    bird_mean, bird_var = tf.nn.moments(output_bird,axes=[0])
    bird_var = tf.cond(tf.equal(tf.is_nan(bird_var),False),lambda:tf.identity(bird_var),lambda:tf.identity(0.))
    #bird_var = tf.sqrt(bird_var)

    labels_boat = tf.equal(labels_gather,4)
    pixel_inds_boat = tf.squeeze(tf.where(labels_boat), 1)
    output_boat = tf.gather(output_gather, pixel_inds_boat)
    output_boat = output_boat[:,4]
    boat_mean, boat_var = tf.nn.moments(output_boat,axes=[0])
    boat_var = tf.cond(tf.equal(tf.is_nan(boat_var),False),lambda:tf.identity(boat_var),lambda:tf.identity(0.))
    #boat_var = tf.sqrt(boat_var)

    labels_bottle = tf.equal(labels_gather,5)
    pixel_inds_bottle = tf.squeeze(tf.where(labels_bottle), 1)
    output_bottle = tf.gather(output_gather, pixel_inds_bottle)
    output_bottle = output_bottle[:,5]
    bottle_mean, bottle_var = tf.nn.moments(output_bottle,axes=[0])
    bottle_var = tf.cond(tf.equal(tf.is_nan(bottle_var),False),lambda:tf.identity(bottle_var),lambda:tf.identity(0.))
    #bottle_var = tf.sqrt(bottle_var)

    labels_bus = tf.equal(labels_gather,6)
    pixel_inds_bus = tf.squeeze(tf.where(labels_bus), 1)
    output_bus = tf.gather(output_gather, pixel_inds_bus)
    output_bus = output_bus[:,6]
    bus_mean, bus_var = tf.nn.moments(output_bus,axes=[0])
    bus_var = tf.cond(tf.equal(tf.is_nan(bus_var),False),lambda:tf.identity(bus_var),lambda:tf.identity(0.))
    #bus_var = tf.sqrt(bus_var)

    labels_car = tf.equal(labels_gather,7)
    pixel_inds_car = tf.squeeze(tf.where(labels_car), 1)
    output_car = tf.gather(output_gather, pixel_inds_car)
    output_car = output_car[:,7]
    car_mean, car_var = tf.nn.moments(output_car,axes=[0])
    car_var = tf.cond(tf.equal(tf.is_nan(car_var),False),lambda:tf.identity(car_var),lambda:tf.identity(0.))
    #car_var = tf.sqrt(car_var)

    labels_cat = tf.equal(labels_gather,8)
    pixel_inds_cat = tf.squeeze(tf.where(labels_cat), 1)
    output_cat = tf.gather(output_gather, pixel_inds_cat)
    output_cat = output_cat[:,8]
    cat_mean, cat_var = tf.nn.moments(output_cat,axes=[0])
    cat_var = tf.cond(tf.equal(tf.is_nan(cat_var),False),lambda:tf.identity(cat_var),lambda:tf.identity(0.))
    #cat_var = tf.sqrt(cat_var)

    labels_chair = tf.equal(labels_gather,9)
    pixel_inds_chair = tf.squeeze(tf.where(labels_chair), 1)
    output_chair = tf.gather(output_gather, pixel_inds_chair)
    output_chair = output_chair[:,9]
    chair_mean, chair_var = tf.nn.moments(output_chair,axes=[0])
    chair_var = tf.cond(tf.equal(tf.is_nan(chair_var),False),lambda:tf.identity(chair_var),lambda:tf.identity(0.))
    #chair_var = tf.sqrt(chair_var)

    labels_cow = tf.equal(labels_gather,10)
    pixel_inds_cow = tf.squeeze(tf.where(labels_cow), 1)
    output_cow = tf.gather(output_gather, pixel_inds_cow)
    output_cow = output_cow[:,10]
    cow_mean, cow_var = tf.nn.moments(output_cow,axes=[0])
    cow_var = tf.cond(tf.equal(tf.is_nan(cow_var),False),lambda:tf.identity(cow_var),lambda:tf.identity(0.))
    #cow_var = tf.sqrt(cow_var)

    labels_table = tf.equal(labels_gather,11)
    pixel_inds_table = tf.squeeze(tf.where(labels_table), 1)
    output_table = tf.gather(output_gather, pixel_inds_table)
    output_table = output_table[:,11]
    table_mean, table_var = tf.nn.moments(output_table,axes=[0])
    table_var = tf.cond(tf.equal(tf.is_nan(table_var),False),lambda:tf.identity(table_var),lambda:tf.identity(0.))
    #table_var = tf.sqrt(table_var)

    labels_dog = tf.equal(labels_gather,12)
    pixel_inds_dog = tf.squeeze(tf.where(labels_dog), 1)
    output_dog = tf.gather(output_gather, pixel_inds_dog)
    output_dog = output_dog[:,12]
    dog_mean, dog_var = tf.nn.moments(output_dog,axes=[0])
    dog_var = tf.cond(tf.equal(tf.is_nan(dog_var),False),lambda:tf.identity(dog_var),lambda:tf.identity(0.))
    #dog_var = tf.sqrt(dog_var)

    labels_horse = tf.equal(labels_gather,13)
    pixel_inds_horse = tf.squeeze(tf.where(labels_horse), 1)
    output_horse = tf.gather(output_gather, pixel_inds_horse)
    output_horse = output_horse[:,13]
    horse_mean, horse_var = tf.nn.moments(output_horse,axes=[0])
    horse_var = tf.cond(tf.equal(tf.is_nan(horse_var),False),lambda:tf.identity(horse_var),lambda:tf.identity(0.))
    #horse_var = tf.sqrt(horse_var)

    labels_motorbike = tf.equal(labels_gather,14)
    pixel_inds_motorbike = tf.squeeze(tf.where(labels_motorbike), 1)
    output_motorbike = tf.gather(output_gather, pixel_inds_motorbike)
    output_motorbike = output_motorbike[:,14]
    motorbike_mean, motorbike_var = tf.nn.moments(output_motorbike,axes=[0])
    motorbike_var = tf.cond(tf.equal(tf.is_nan(motorbike_var),False),lambda:tf.identity(motorbike_var),lambda:tf.identity(0.))
    #motorbike_var = tf.sqrt(motorbike_var)

    labels_person = tf.equal(labels_gather,15)
    pixel_inds_person = tf.squeeze(tf.where(labels_person), 1)
    output_person = tf.gather(output_gather, pixel_inds_person)
    output_person = output_person[:,15]
    person_mean, person_var = tf.nn.moments(output_person,axes=[0])
    person_var = tf.cond(tf.equal(tf.is_nan(person_var),False),lambda:tf.identity(person_var),lambda:tf.identity(0.))
    #person_var = tf.sqrt(person_var)

    labels_plant = tf.equal(labels_gather,16)
    pixel_inds_plant = tf.squeeze(tf.where(labels_plant), 1)
    output_plant = tf.gather(output_gather, pixel_inds_plant)
    output_plant = output_plant[:,16]
    plant_mean, plant_var = tf.nn.moments(output_plant,axes=[0])
    plant_var = tf.cond(tf.equal(tf.is_nan(plant_var),False),lambda:tf.identity(plant_var),lambda:tf.identity(0.))
    #plant_var = tf.sqrt(plant_var)

    labels_sheep = tf.equal(labels_gather,17)
    pixel_inds_sheep = tf.squeeze(tf.where(labels_sheep), 1)
    output_sheep = tf.gather(output_gather, pixel_inds_sheep)
    output_sheep = output_sheep[:,17]
    sheep_mean, sheep_var = tf.nn.moments(output_sheep,axes=[0])
    sheep_var = tf.cond(tf.equal(tf.is_nan(sheep_var),False),lambda:tf.identity(sheep_var),lambda:tf.identity(0.))
    #sheep_var = tf.sqrt(sheep_var)

    labels_sofa = tf.equal(labels_gather,18)
    pixel_inds_sofa = tf.squeeze(tf.where(labels_sofa), 1)
    output_sofa = tf.gather(output_gather, pixel_inds_sofa)
    output_sofa = output_sofa[:,18]
    sofa_mean, sofa_var = tf.nn.moments(output_sofa,axes=[0])
    sofa_var = tf.cond(tf.equal(tf.is_nan(sofa_var),False),lambda:tf.identity(sofa_var),lambda:tf.identity(0.))
    #sofa_var = tf.sqrt(sofa_var)

    labels_train = tf.equal(labels_gather,19)
    pixel_inds_train = tf.squeeze(tf.where(labels_train), 1)
    output_train = tf.gather(output_gather, pixel_inds_train)
    output_train = output_train[:,19]
    train_mean, train_var = tf.nn.moments(output_train,axes=[0])
    train_var = tf.cond(tf.equal(tf.is_nan(train_var),False),lambda:tf.identity(train_var),lambda:tf.identity(0.))
    #train_var = tf.sqrt(train_var)

    labels_tv= tf.equal(labels_gather,20)
    pixel_inds_tv = tf.squeeze(tf.where(labels_tv), 1)
    output_tv = tf.gather(output_gather, pixel_inds_tv)
    output_tv = output_tv[:,20]
    tv_mean, tv_var = tf.nn.moments(output_tv,axes=[0])
    tv_var = tf.cond(tf.equal(tf.is_nan(tv_var),False),lambda:tf.identity(tv_var),lambda:tf.identity(0.))
    #tv_var = tf.sqrt(tv_var)
    intra_class_loss = plane_abs
    #intra_class_loss = 0.05*tf.add_n([plane_var, bike_var, bird_var, boat_var, bottle_var, bus_var, car_var, cat_var, chair_var, cow_var,
    	#                  table_var, dog_var, horse_var, motorbike_var, person_var, plant_var, sheep_var, sofa_var, train_var, tv_var])
    final_loss = tf.reduce_mean(loss_normal)+intra_class_loss+inter_class_loss
    return final_loss,intra_class_loss,inter_class_loss
