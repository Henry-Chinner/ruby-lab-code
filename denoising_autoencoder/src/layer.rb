require './data_loader.rb'
require './nn_functions.rb'
require 'memoist'
require 'nmatrix'
require './vec_to_image.rb'


class Layer
  include NNFunctions
  extend Memoist

  attr_accessor :w, :z, :a, :a_with_bias, :grad, :delta, :y, :weight_decay, :y_pointer, :weight_update_status, :corruption_level , :w_accum , :momentum
  attr_reader   :a_func, :fan_in, :fan_out, :name, :x

  def initialize(opts = {})
    @name = opts[:name]
    @a_func = opts[:a_func]
    @fan_in = opts[:fan_in]
    @fan_out = opts[:fan_out]
    @weight_decay = opts[:weight_decay]
    @y_pointer = opts[:y_pointer]
    @weight_update_status = opts[:weight_update_status]
    @corruption_level = opts[:corruption_level]
    @with_momentum = opts[:with_momentum]
    @momentum = opts[:momentum]
    @z = []
    @a = []
    @a_with_bias = []
    @grad = []
    @delta = []
    @w = initialize_new_weight
    @w_accum = NMatrix.zeroes(@w.shape)
  end
  
  def initialize_new_weight
    if @a_func[:name] == 'softmax'
      init_factor = 0.1 / ((@fan_in + 1) ** (0.5))    
      @w = (NMatrix.random([@fan_in + 1,@fan_out],:dtype => :float32) - (NMatrix.ones([@fan_in + 1,@fan_out],:dtype => :float32) /2)) * init_factor 
    elsif @a_func[:name] == 'sin'
      init_factor = 10.1 / ((@fan_in + 1) ** (0.5))    
      @w = (NMatrix.random([@fan_in + 1,@fan_out],:dtype => :float32) - (NMatrix.ones([@fan_in + 1,@fan_out],:dtype => :float32) /2)) * init_factor 
    else
      init_factor = 0.01 / ((@fan_in + 1) ** (0.5))    
      @w = (NMatrix.random([@fan_in + 1,@fan_out],:dtype => :float32) - (NMatrix.ones([@fan_in + 1,@fan_out],:dtype => :float32) /2)) * init_factor  
    end  
  end

  def update_weights(alpha)
    if @weight_update_status == true
      if @weight_decay != 0
        if @with_momentum == true
          @w_accum[0..(@fan_in-1),0..(@fan_out-1)] = (@w_accum[0..(@fan_in-1),0..(@fan_out-1)] * @momentum) + (@w[0..(@fan_in-1),0..(@fan_out-1)] * @weight_decay + @delta[0..(@fan_in-1),0..(@fan_out-1)]) * alpha
          @w[0..(@fan_in-1),0..(@fan_out-1)] = @w[0..(@fan_in-1),0..(@fan_out-1)] - @w_accum[0..(@fan_in-1),0..(@fan_out-1)]
        else
          @w[0..(@fan_in-1),0..(@fan_out-1)] = @w[0..(@fan_in-1),0..(@fan_out-1)] * (1-@weight_decay*alpha) - @delta[0..(@fan_in-1),0..(@fan_out-1)] * alpha
        end  
      else
        if @with_momentum == true
          @w_accum[0..(@fan_in-1),0..(@fan_out-1)] = (@w_accum[0..(@fan_in-1),0..(@fan_out-1)] * @momentum) + @delta[0..(@fan_in-1),0..(@fan_out-1)] * alpha
          @w[0..(@fan_in-1),0..(@fan_out-1)] = @w[0..(@fan_in-1),0..(@fan_out-1)] - @w_accum[0..(@fan_in-1),0..(@fan_out-1)]
        else
          @w[0..(@fan_in-1),0..(@fan_out-1)] = @w[0..(@fan_in-1),0..(@fan_out-1)] - @delta[0..(@fan_in-1),0..(@fan_out-1)] * alpha
        end  
      end

      if @with_momentum == true 
        @w_accum[@fan_in,0..(@fan_out-1)] = @w_accum[@fan_in,0..(@fan_out-1)] * @momentum + @delta[@fan_in,0..(@fan_out-1)] * alpha 
        @w[@fan_in,0..(@fan_out-1)] = @w[@fan_in,0..(@fan_out-1)] -  @w_accum[@fan_in,0..(@fan_out-1)] * alpha #bias weights
      else
        @w[@fan_in,0..(@fan_out-1)] = @w[@fan_in,0..(@fan_out-1)] - @delta[@fan_in,0..(@fan_out-1)] * alpha #bias weights
      end  
    end
  end

  def viz_weights
    VecToImage.viz_hidden_layer(15,@fan_out / 15,@w[0..(@fan_in-1),0..(@fan_out-1)].transpose.to_a)
  end   

  def scalar_mat(dim,scalar)
    NMatrix.eye(dim) * scalar
  end

  memoize :scalar_mat 

  def error
    if @a_func[:name] == 'relu'
       @grad = ((@y-@a) *  derivative(@z.clone,@a_func,false)) * -1
    else
      @grad = -(@y-@a)
    end
  end

  def get_classification
    @a.each_with_index.max[1]  
  end 

  def calc_activation
    if @corruption_level.nil?
      @a = activation_function(@z.clone,@a_func)
      @y = @a
    else
      @a = activation_function(@z.clone,@a_func)
      @y = @a.clone
      @a = add_noise(@a,@corruption_level)
    end      
  end

  def calc_derivative
    if @a_func[:name] == ('tanh' or 'sigmoid')
      derivative(@a.clone,@a_func,true)
    else
      derivative(@z.clone,@a_func,false)
    end  
  end

  def add_bias
    @a_with_bias = shift_mat(@a)
  end

  def wake_up
     @w = @w.to_nm
  end 
  
  def prep_persist
    @w = @w.to_a
    @w_accum = []
    @z = []
    @a = []
    @a_with_bias = []
    @y = []
    @grad = []
    @delta = []
  end
    
end

class DataLayer 
  include NNFunctions

  attr_accessor :z, :a, :a_with_bias, :batch_size, :corruption_level ,:dt , :mode, :data_cursor
  attr_reader   :fan_out, :name , :a_func ,:y 

  def initialize(opts = {})
    @data_file = opts[:data_file]
    @name = opts[:name]
    @a_func = opts[:a_func]   
    @fan_out = opts[:fan_out]
    @batch_size = opts[:batch_size]
    @corruption_level = opts[:corruption_level]
    @mode = opts[:mode]
    @a = []
    @z = []
    @a_with_bias = []
    @y = []
    load_data(@data_file,label_index = :none)
  end

  def load_data(data_file = @data_file,label_index = :none)
    begin
      @dt = DataTable.load("../data/#{data_file}.data")
    rescue
      puts "Loading file from disk"          
      @dt = DataTable.new({:file => "../data/#{data_file}.csv" , :label_index => label_index})
      @dt.persist("../data/#{data_file}.data")
    end
    shuffle_data
    set_data_cursor
  end

  def update_weights(alpha)
    
  end

  def update_bias_weights(alpha)
   
  end


  def set_data_cursor
    @data_cursor = @dt.observations.each_slice(@batch_size)
  end  

  def shuffle_data
    @dt.observations.shuffle!
  end   

  def pull_z
    begin
        obs = @data_cursor.next
        if obs.size != @batch_size
          set_data_cursor
          obs = @data_cursor.next
        end  
    rescue =>e 
      if e.message == 'iteration reached an end' and (@mode == 'train' or @mode =='pretrain')
        set_data_cursor
        obs = @data_cursor.next
      elsif e.message == 'iteration reached an end' and @mode == 'test'
        puts e.message
      end  
    end  
      
    @z = obs.map{|el| el.features}.flatten.to_nm([obs.size,@fan_out])
    @y = NMatrix.zeroes([obs.size,10])

    if @mode == 'train'
      obs.each_with_index{ |el,i| @y[i,el.label] = 1.0}  
    end
  end

  

  def add_bias
    @a_with_bias = shift_mat(@a) 
  end   

  def calc_activation
    if @corruption_level.nil?
      @a = activation_function(@z,@a_func)
    else
      @a = activation_function(@z,@a_func)
      @y = @a.clone
      @a = add_noise(@a,@corruption_level)
    end      
  end

  def wake_up
    load_data
  end  

  def prep_persist
    @dt = []
    @z = []
    @a = []
    @a_with_bias = []
    @y = []
    @data_cursor = []
  end
  
end  