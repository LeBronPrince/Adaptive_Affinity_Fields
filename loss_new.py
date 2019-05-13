labels_chair = tf.equal(labels_gather,9)
pixel_inds_chair = tf.squeeze(tf.where(labels_chair), 1)
labels_chair_gather = tf.to_int32(tf.gather(labels_gather, pixel_inds_chair))
output_chair = output_gather[:,:,:,9]
chair_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_chair_gather,logits=output_chair)
chair_loss = tf.reduce_mean(chair_loss)
