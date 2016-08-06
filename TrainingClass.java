public class TrainingClass {
    static void training() {
        int count_learn;
        if (my_keyboard.fast_learning) count_learn = flat_agent.batch_size;
        else count_learn = 10;

        float [][] output = new float[count_learn][a0_network.netStructure[3]];
        float [][] input = new float[count_learn][a0_network.netStructure[0]];

        System.out.println("batch starting ("+count_learn+")...");
        for (int i = 0; i < count_learn; i++) {

            int ii = functions.closer_to_end(pap, flat_agent.D.size());

            if (flat_agent.D.get(ii).final_state) //ean eimaste stin teliki thesi/adieksodo
            {
                for (int iii = 0; iii < output[i].length; iii++) output[i][iii] = flat_agent.D.get(ii).get_value;
            }
            else
            {
                float[] new_state = flat_agent.D.get(ii).f_new;

                INDArray input_new = Nd4j.create(new_state, new int[] { 1, a0_network.netStructure[0] });
                INDArray New_state_all_actions_INDArray = flat_agent.neuralnet2.output(input_new, false);
                float [] New_state_all_actions = new float[flat_agent.a];


                for (int iii = 0; iii < flat_agent.a; iii++) {
                    if (!flat_agent.D.get(ii).possible_action[iii]) New_state_all_actions[iii] = -10;
                    else New_state_all_actions[iii]=New_state_all_actions_INDArray.getFloat(iii);
                }
                float Qf_new_best = pap.max(New_state_all_actions);

                float[] current_state = flat_agent.D.get(ii).f_old;

                INDArray input_temp = Nd4j.create(current_state, new int[] { 1, a0_network.netStructure[0] });
                INDArray response = flat_agent.neuralnet2.output(input_temp, false);

                for (int iii = 0; iii < output[i].length; iii++)   //gia kathe output
                {
                    if (flat_agent.D.get(ii).act == iii) output[i][iii] = flat_agent.D.get(ii).get_value + flat_agent.gama * Qf_new_best;
                    else output[i][iii] = response.getFloat(iii);
                }
            }
            input[i] = flat_agent.D.get(ii).f_old;
        }

        INDArray inputNDArray_alex = Nd4j.create(input);//,new int[]{1, a0_network.netStructure[0]});
        INDArray outputNDArray_alex = Nd4j.create(output);//, new int[]{1, 4});
        DataSet dataSet = new DataSet(inputNDArray_alex, outputNDArray_alex);

        System.out.println("training starting ("+count_learn+")...");
        flat_agent.neuralnet2.fit(dataSet);
        System.out.println("...training finished!!!");
    }
}
