public class a0_network {

    static float LEARNING_RATE = 0.01f;
    public static int netStructure[] = {100, 1, 500, 4};


    static void initialize_network(PApplet pap) {
        sigmoid.setupSigmoid();

        flat_agent.neuralnet = new network.Network(pap,
            netStructure[0],
            netStructure[1],
            netStructure[2],
            netStructure[3]);
    }


    public static final int seed = 12345;

    public static final int iterations = 5;

    public static final int nEpochs = 1;


    public static final Random rng = new Random(seed);

    static void initialize_network2(PApplet pap) {
        System.out.println("seed: "+rng.nextInt());
        //DL4J -- Create the network
        int numInput = 100;//netStructure[0]; //75 //TODO: 24/07/16 to net Structure[1] na paei sto DL4J
        int numOutputs = netStructure[3];
        int nHidden = netStructure[2];
        int nChannels = flat_agent.rooms_types; //todo na pairnei ton arithmo apo to netStructure[3] = ta diafora actions pou tha kanei



        //Convolutional network
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
            .seed(rng.nextInt())
            .iterations(iterations)
            //.regularization(true).l2(0.0005)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(LEARNING_RATE)
            .weightInit(WeightInit.XAVIER)
            //.updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new ConvolutionLayer.Builder(3, 3)
                .nIn(nChannels)
                .stride(1, 1)
//                .padding(1,1)
                .nOut(20)//number of filters
                .activation("relu")
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .padding(1,1)
//                .kernelSize(2,2)//pooling
                .stride(1,1)
                .build())
            .layer(2, new DenseLayer.Builder().activation("relu")
                .nOut(500).build())
            .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)

                .nOut(numOutputs)
                .activation("identity")//softmax
                .build())
            .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,5,5,nChannels);
        MultiLayerConfiguration conf = builder.build();
        flat_agent.neuralnet2 = new MultiLayerNetwork(conf);


        //initialization
        flat_agent.neuralnet2.init();
        flat_agent.neuralnet2.setListeners(new ScoreIterationListener(1));
    }
}
