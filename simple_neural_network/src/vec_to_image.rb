require 'chunky_png'
require './data_loader.rb'

class VectoImage

	def initialize(pixels_wide,pixels_high)
	@dt = DataTable.load('../data/train.data')
	@images_wide = pixels_wide / 28
	@images_high = pixels_high /28
	
	image = ChunkyPNG::Image.new(@images_wide * 28,@images_high * 28)
	
	@images_wide.times do |i|
		puts i
		@images_high.times do |j|
			puts j

			# s1 = rand
			# s2 = (1-s1) * rand
			# s3 = (1-s1-s2)
		small_image = @dt.sample.features
			small_image.size.times do |k|
				#colored version -- image[(i)*(28) + (k % 28),(j)*(28) + (k / 28)] = ChunkyPNG::Color.rgb((small_image[k] * s2).ceil,(small_image[k] * s3).ceil,(small_image[k] * s1).ceil)
				image[(i)*(28) + (k % 28),(j)*(28) + (k / 28)] = ChunkyPNG::Color.grayscale(small_image[k])
			
			end
		end
	end

	image.save('mnist.png')

	end	

end

i = VectoImage.new(2500,1200)

