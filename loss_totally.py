
def loss_total(output_gather,labels_gather):

    loss_normal = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_gather, labels=labels_gather)
    ###chair
    labels_chair = tf.equal(labels_gather,9)
    pixel_inds_chair = tf.squeeze(tf.where(labels_chair), 1)
    labels_chair_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_chair))
    output_chair = tf.gather(output_gather, pixel_inds_chair)
    assert output_chair.get_shape().as_list()[0] == labels_chair_gather.get_shape().as_list()[0]
    output_chair = tf.nn.softmax(output_chair)
    output_chair_table = tf.identity(output_chair[:,9])
    output_chair_table_loss = -tf.log(output_chair_table)
    loss_chair_table = tf.reduce_mean(output_chair_table_loss)
    loss_chair_table = tf.cond(tf.equal(tf.is_nan(loss_chair_table),False),lambda:tf.identity(loss_chair_table),lambda:tf.identity(0.))
    chair_loss = loss_chair_table

    ###table
    labels_table = tf.equal(labels_gather,11)
    pixel_inds_table = tf.squeeze(tf.where(labels_table), 1)
    labels_table_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_table))
    output_table = tf.gather(output_gather, pixel_inds_table)
    assert output_table.get_shape().as_list()[0] == labels_table_gather.get_shape().as_list()[0]
    output_table = tf.nn.softmax(output_table)
    output_table_chair = tf.identity(output_table[:,11])
    output_table_chair = tf.log(output_table_chair)
    loss_table_chair = -tf.reduce_mean(output_table_chair)
    loss_table_chair = tf.cond(tf.equal(tf.is_nan(loss_table_chair),False),lambda:tf.identity(loss_table_chair),lambda:tf.identity(0.))
    table_loss = loss_table_chair


    ###bike
    labels_bike = tf.equal(labels_gather,2)
    pixel_inds_bike = tf.squeeze(tf.where(labels_bike), 1)
    labels_bike_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_bike))
    output_bike = tf.gather(output_gather, pixel_inds_bike)
    assert output_bike.get_shape().as_list()[0] == labels_bike_gather.get_shape().as_list()[0]
    output_bike = tf.nn.softmax(output_bike)
    output_bike_bike = tf.identity(output_bike[:,2])
    output_bike_bike = tf.log(output_bike_bike)
    loss_bike_bike = -tf.reduce_mean(output_bike_bike)
    loss_bike_bike = tf.cond(tf.equal(tf.is_nan(loss_bike_bike),False),lambda:tf.identity(loss_bike_bike),lambda:tf.identity(0.))
    bike_loss = loss_bike_bike
    bike_loss = loss_bike_bike


    ###bottle
    labels_bottle = tf.equal(labels_gather,5)
    pixel_inds_bottle = tf.squeeze(tf.where(labels_bottle), 1)
    labels_bottle_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_bottle))
    output_bottle = tf.gather(output_gather, pixel_inds_bottle)
    assert output_bottle.get_shape().as_list()[0] == labels_bottle_gather.get_shape().as_list()[0]
    output_bottle = tf.nn.softmax(output_bottle)
    output_bottle_bottle = tf.identity(output_bottle[:,5])
    output_bottle_bottle = tf.log(output_bottle_bottle)
    loss_bottle_bottle = -tf.reduce_mean(output_bottle_bottle)
    loss_bottle_bottle = tf.cond(tf.equal(tf.is_nan(loss_bottle_bottle),False),lambda:tf.identity(loss_bottle_bottle),lambda:tf.identity(0.))
    bottle_loss = loss_bottle_bottle
    ###boat
    labels_boat = tf.equal(labels_gather,4)
    pixel_inds_boat = tf.squeeze(tf.where(labels_boat), 1)
    labels_boat_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_boat))
    output_boat = tf.gather(output_gather, pixel_inds_boat)
    assert output_boat.get_shape().as_list()[0] == labels_boat_gather.get_shape().as_list()[0]
    output_boat = tf.nn.softmax(output_boat)
    output_boat_boat = tf.identity(output_boat[:,4])
    output_boat_boat = tf.log(output_boat_boat)
    loss_boat_boat = -tf.reduce_mean(output_boat_boat)
    loss_boat_boat = tf.cond(tf.equal(tf.is_nan(loss_boat_boat),False),lambda:tf.identity(loss_boat_boat),lambda:tf.identity(0.))
    boat_loss = loss_boat_boat

    ###plant
    labels_plant = tf.equal(labels_gather,16)
    pixel_inds_plant = tf.squeeze(tf.where(labels_plant), 1)
    labels_plant_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_plant))
    output_plant = tf.gather(output_gather, pixel_inds_plant)
    assert output_plant.get_shape().as_list()[0] == labels_plant_gather.get_shape().as_list()[0]
    output_plant = tf.nn.softmax(output_plant)
    output_plant_plant = tf.identity(output_plant[:,16])
    output_plant_plant = tf.log(output_plant_plant)
    loss_plant_plant = -tf.reduce_mean(output_plant_plant)
    loss_plant_plant = tf.cond(tf.equal(tf.is_nan(loss_plant_plant),False),lambda:tf.identity(loss_plant_plant),lambda:tf.identity(0.))
    plant_loss = loss_plant_plant

    ###sofa
    labels_sofa = tf.equal(labels_gather,18)
    pixel_inds_sofa = tf.squeeze(tf.where(labels_sofa), 1)
    labels_sofa_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_sofa))
    output_sofa = tf.gather(output_gather, pixel_inds_sofa)
    assert output_sofa.get_shape().as_list()[0] == labels_sofa_gather.get_shape().as_list()[0]
    output_sofa = tf.nn.softmax(output_sofa)

    output_sofa_sofa = tf.identity(output_sofa[:,18])
    output_sofa_sofa_loss = -tf.log(output_sofa_sofa)
    loss_sofa_sofa = tf.reduce_mean(output_sofa_sofa_loss)
    loss_sofa_sofa = tf.cond(tf.equal(tf.is_nan(loss_sofa_sofa),False),lambda:tf.identity(loss_sofa_sofa),lambda:tf.identity(0.))
    sofa_loss = loss_sofa_sofa

    ###tv
    labels_tv = tf.equal(labels_gather,20)
    pixel_inds_tv = tf.squeeze(tf.where(labels_tv), 1)
    labels_tv_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_tv))
    output_tv = tf.gather(output_gather, pixel_inds_tv)
    assert output_tv.get_shape().as_list()[0] == labels_tv_gather.get_shape().as_list()[0]
    output_tv = tf.nn.softmax(output_tv)
    output_tv_tv = tf.identity(output_tv[:,20])
    output_tv_tv = tf.log(output_tv_tv)
    loss_tv_tv = -tf.reduce_mean(output_tv_tv)
    loss_tv_tv = tf.cond(tf.equal(tf.is_nan(loss_tv_tv),False),lambda:tf.identity(loss_tv_tv),lambda:tf.identity(0.))
    tv_loss = loss_tv_tv
    inter_class_loss = tf.add_n([chair_loss,table_loss,bike_loss,boat_loss,plant_loss,sofa_loss,tv_loss,bottle_loss])
    inter_class_loss = 0.1*inter_class_loss

    final_loss = tf.reduce_mean(loss_normal)+inter_class_loss
    return final_loss,inter_class_loss
