package com.felixgrund.codeshovel.changes;

import com.felixgrund.codeshovel.wrappers.StartEnvironment;
import com.felixgrund.codeshovel.parser.Yfunction;
import com.google.gson.JsonObject;

public class Yannotationchange extends Ysignaturechange {

    public Yannotationchange(StartEnvironment startEnv, Yfunction newFunction, Yfunction oldFunction) {
        super(startEnv, newFunction, oldFunction);
    }

    @Override
    protected Object getOldValue() {
        return oldFunction.getAnnotation();
    }

    @Override
    protected Object getNewValue() {
        return newFunction.getAnnotation();
    }


}
