require './data_loader.rb'
require 'nmatrix'
require 'csv'
require 'optparse'


class NeuralNet

  def initialize(opts ={})
  puts_hrow

      
  @hidden_func = opts[:hidden_func]
  @output_func = opts[:output_func]
  @eval_or_train = opts[:mode]
  

  @number_of_classes = 10
  @input_size = 784

    if @eval_or_train == 'train'

      begin
        @dt = DataTable.load('../data/train.data')
      rescue
        puts "Loading file from disk"
        puts_hrow            
        @dt = DataTable.new({:file => '../data/train.csv' , :label_index => 0})
        @dt.persist('../data/train.data')
      end
      @hidden_nodes = opts[:hidden_nodes] 
      @error_history = []
      @classification_history = []
      @alpha = opts[:alpha]

      self.train

    elsif @eval_or_train == 'eval'

      self.create_test_submission

    else
     puts "You have to set --mode to 'train' or 'eval'"    
    end    


    
  end


  def initialize_new_weights
    @init_factor_1 = 0.01 / ((@input_size + 1) ** (0.5))    # weight scaler for the first weight matrix. It is indirectly proportional to the input size. It has to be, otherwise you will blow up your hidden layer. If you do not put in the deflator. The first couple of itterations will work on large sums in the hidden layer. Large sums are bad in neural networks.
    @init_factor_2 = 0.01 / (@hidden_nodes ** (0.5))      
    
    @w1 = (NMatrix.random([@input_size + 1,@hidden_nodes]) - (NMatrix.ones([@input_size + 1,@hidden_nodes]) /2)) * @init_factor_1 # some basic matrix algebra, create a matrix with 
    @w2 = (NMatrix.random([@hidden_nodes + 1,@number_of_classes])  - (NMatrix.ones([@hidden_nodes + 1,@number_of_classes]) / 2)) * @init_factor_2
  end
  
  def load_trained_weights
    @w1 = Marshal.load(File.binread('../data/w1.txt')).to_nm
    @w2 = Marshal.load(File.binread('../data/w2.txt')).to_nm
  end

  def create_test_submission
   puts "Creating submission" 
   load_trained_weights
   @hidden_nodes = @w1.shape[1] 

    data_table = DataTable.new({:file => '../data/test.csv' , :label_index => :none})

    CSV.open("../data/submission.csv", "wb") do |csv|

      csv << ["ImageID", "Label"]

      data_table.observations.each_with_index do |observation,i |
      
          csv << [(i+1).to_s, forward(observation,{:eval_or_train => 'eval'}) ]
      end
    end

  end 

def forward(observation,opts ={})
  #convert the features array into a NMatrix matrix and divide every element by 255. 
  #the division scales down the input.The input vector is initialized with size 
  #1 bigger than the @input_size. This is to accommodate the bias term 
  a1 = observation.features.flatten.to_nm([1,@input_size + 1]) / 255.0

  #Set the bias term equal to 1
  #NMatrix 2 dimensional matrices can be accessed via [row,column]
  a1[0, @input_size ] = 1.0

  #pass the product of the input values and the arc weight forward 
  #and sum the product up at each node
  
  z2 = a1.dot(@w1)

  #apply the activation function to the sum vector element wise
  a2 = activation_function(z2,@hidden_func)

  #resize the hidden layer to add the bias unit
  a2_with_bias = NMatrix.zeroes([1,@hidden_nodes+1])
  a2_with_bias[0,0..@hidden_nodes] = a2
  a2_with_bias[0,@hidden_nodes] = 1.0 

  #z3 = a2 x @w2, propogating the hidden layer forward to get the sums in the output layer

  z3 = a2_with_bias.dot(@w2)
 
  #Softmax activation function in the output layer
  a3 = activation_function(z3,@output_func)
  
  #if in training mode, pass values of layers to backprop. 
  #otherwise return the prediction the output layer
  if opts[:eval_or_train] == 'train'
    backprop(a1,a2_with_bias,z2,z3,a3,observation.label)
  elsif opts[:eval_or_train] == 'eval'
    return a3.each_with_index.max[1]
  end 
    
end
  
def backprop(a1,a2_with_bias,z2,z3,a3,label)
  #initiates the output vector of zeroes
  y = NMatrix.zeroes([1,10])

  #set the label from the data to 1
  #only 1 element can be 1 at a time as classes
  #are mutually exclusive
  y[0,label] = 1.0
  
  #derivative of the loss function. Difference between predicted
  #values and the true value  
  
  d3 = -(y - a3)  

  #using the derivative d3 is a good enough measure to 
  #see if the cost is decreasing so we append it to
  #the error history
  @error_history <<  d3.transpose.abs.sum[0]
  
  #add 1 to the classification history if the prediction
  #is correct, otherwise zero
  @classification_history <<  (a3.each_with_index.max[1] == label ? 1.0 : 0.0)
    
  # derivative, has the same size as the hidden layer. The range [] operator
  # excludes the bias node. No error is passed to the bias node.  
  d2 =  @w2.dot( d3.transpose )[0..(@hidden_nodes-1)] * derivative(z2.transpose,@hidden_func)    
  
  #matrix with dimensions equal to @w1's dimensions
  #each element contains the gradient of the weight
  #with respect to the cost function. If the weights
  #are reduced by a small fraction of this value the cost function
  #will go down     
  grad1 = d2.dot(a1)
  
  #same for @w2 
  grad2 = d3.transpose.dot(a2_with_bias)

  # updating the weigh matrices. The first layer is updated
  # by a factor of 10 less than than the second layer. for numerical
  # stability. Big weight changes -> big weights -> equals big sums -> saturated neurons
    
  @w1 = @w1 - grad1.transpose * @alpha  * 0.1         
  @w2 = @w2 - grad2.transpose * @alpha  
end

def train
  puts "Entered Training"
  i = 0
  start_time = Time.now
  initialize_new_weights
    
  loop do 
    # forward pass in the network with a random observation from @dt.sample.
    # eval_or_train is passed as train becuase the forward method has to pass
    # it's results to the backprop method. The backprop method will update the weights
    forward(@dt.sample,{:eval_or_train => 'train'})

    ave_error_history = running_average(1000,@error_history)
    ave_error_history_5000 = running_average(5000,@error_history)
    ave_classification_history = running_average(1000,@classification_history)
    ave_classification_history_5000 = running_average(5000,@classification_history)
    ratio = (ave_classification_history  / ave_classification_history_5000)

    puts "Running Average Error (1000) => #{ave_error_history}"
    puts "Running Average Error (5000) => #{ave_error_history_5000}"
    puts "Running Average Classification (1000) => #{ave_classification_history} "
    puts "Running Average Classification (5000) => #{ave_classification_history_5000}"
    puts "Classification Runninge Average Ratio => #{ratio}"
  
    puts "Iteration = #{i}"
    puts "---"

    if ratio < 1.0 and i > 60000
      finish_time = Time.now
      puts File.open('../data/w1.txt','w'){|f| f << Marshal.dump(@w1.to_a)}
      puts File.open('../data/w2.txt','w'){|f| f << Marshal.dump(@w2.to_a)}
      puts "Total training time was: #{(finish_time - start_time).round(0)} sec"
      break
    end

  i += 1
  end
end 

  def running_average(scale,ary)
    ary.last(scale).inject{ |sum, el| sum + el}.to_f / [scale,ary.size].min 
  end 

  def sigmoid(mat)
    NMatrix.ones(mat.shape)/(NMatrix.ones(mat.shape)+(-mat).exp)
  end 

  def rect_lin(mat)
    (mat / 10.0).map do |el|
      if el < 0
        0
      else
       el 
      end 

    end 
  end   

  def tanh(mat)
    ( (mat).exp - (-mat).exp )/( (mat).exp + (-mat).exp )
  end

  def sine(mat)
    mat.sin
  end

  def softmax(mat)
    mat.map!{|el| Math::exp(el) }
    sum = mat.inject(0){|sum,el| sum = sum + el}
    mat.map{|el| el / sum.to_f}
  end 

  def activation_function(mat,func)
    if func == 'sin'
      return  mat.sin
    elsif func == 'sigmoid'
      return sigmoid(mat)
    elsif func == 'tanh'
      return tanh(mat)
    elsif func == 'rect_lin'
      return rect_lin(mat)
    elsif func == 'softmax'
      return softmax(mat)         
    end
  end

  def derivative(mat,func)
    if func == 'sin'
      return mat.cos
    elsif func == 'sigmoid'
      return (sigmoid(mat) * (NMatrix.ones(mat.shape) - sigmoid(mat)))
    elsif func == 'tanh'
      return (NMatrix.ones(mat.shape)-tanh(mat)) * (NMatrix.ones(mat.shape) + tanh(mat))
    elsif func == 'rect_lin'
      temp = (mat / 10.0 ).map do |el|
        if el < 0
          0
        else
          1.0 / 10.0
        end 
      end

      return temp      
    end   
  end

  def puts_hrow
    puts "----------------------"   
  end 
end

options = {}

parser = OptionParser.new do|opts|
  opts.banner = "Usage: neural_net.rb [options]"

  opts.on('-a', '--alpha alpha', 'Sets alpha') do |alpha|
    options[:alpha] = alpha.to_f;
  end

  opts.on('--hidden_func func', 'Sets Hidden Function') do |func|
    options[:hidden_func] = func;
  end

  opts.on('--output_func h_func', 'Sets Output Function') do |h_func|
    options[:output_func] = h_func;
  end

  opts.on('--hidden_nodes number', 'Set number of Hidden Nodes') do |number|
    options[:hidden_nodes] = number.to_i;
  end

  opts.on('-m', '--mode mode', 'Mode') do |mode|
    options[:mode] = mode.to_s;
  end

  opts.on('-h', '--help', 'Displays Help') do
    puts opts
    exit
  end

end

parser.parse!

if options[:hidden_nodes].nil?
  options[:hidden_nodes] = 300  
end

if options[:hidden_func].nil?
  options[:hidden_func] = 'tanh'
end

if options[:output_func].nil?
  options[:output_func] = 'softmax'
end

if options[:alpha].nil?
  options[:alpha] = 0.05     
end

if options[:mode].nil? 
  options[:mode] = 'train'    
end


options.each{|k,v| puts "#{k} set to #{v}"}

sleep 2
nn = NeuralNet.new(options)





