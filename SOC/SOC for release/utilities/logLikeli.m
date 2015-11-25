function [valneg, valpos] = logLikeli(err, tao)
    valneg(err<=tao) = -log(tao);
    valneg(err>tao) = log(tao);
    valpos(err>tao) = 0;
    valpos(err<=tao) = log(tao);
end

